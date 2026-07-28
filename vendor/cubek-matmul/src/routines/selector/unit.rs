use std::fmt::Display;

use crate::{
    components::{stage::PartitionBuffering, tile::TileMatmulKind},
    definition::{
        MatmulElems, MatmulGlobalElems, MatmulKind, MatmulProblem, MatmulVectorSizes, SwizzleModes,
        TilingBlueprint, TilingScheme,
    },
};
use cubecl::{
    Runtime,
    client::ComputeClient,
    ir::{StorageType, VectorSize},
};
use cubek_std::{
    MatrixLayout,
    cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint, SmAllocation},
    stage::SwizzleMode,
};

#[derive(Default, Clone, Copy, Debug)]
pub enum TileSizeSelection {
    // Chooses the smallest tile size possible.
    MinTileSize,
    #[default]
    // Chooses the biggest tile size possible.
    MaxTileSize,
}

impl Display for TileSizeSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TileSizeSelection::MinTileSize => f.write_str("min_tile_size"),
            TileSizeSelection::MaxTileSize => f.write_str("max_tile_size"),
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub enum PartitionScaling {
    #[default]
    Enabled,
    Disabled,
}

#[derive(Default, Clone, Copy, Debug)]
pub enum StageScaling {
    Enabled(u8),
    #[default]
    Disabled,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct UnitTilingBlueprintOptions {
    pub tile: TileSizeSelection,
    pub stage: StageScaling,
    pub partition: PartitionScaling,
    pub swizzle: bool,
}

/// [daemon patch — see PATCH.md] Walk a unit stage's `(stage_m, stage_n)` down until the lhs +
/// rhs stages fit the adapter's shared-memory budget. The launch validator refuses a stage set
/// exceeding `hardware.max_shared_memory_size` (e.g. 40,960 B selected vs Apple's 32,768 B limit
/// — tracel-ai/burn#4530, #4851), but no selector consulted the limit (only the gemv setup does);
/// with autotune off there is no fallback and the refusal is fatal. Deterministic per adapter:
/// the walk depends only on the device-property limit. Halves the larger contributor first; if a
/// single tile pair still exceeds the limit, the launch validator keeps the last word.
pub(crate) fn clamp_stage_to_shared_memory(
    t: (u32, u32, u32),
    p: (u32, u32, u32),
    (mut stage_m, mut stage_n): (u32, u32),
    (lhs_stage_size, rhs_stage_size): (usize, usize),
    max_shared: usize,
) -> (u32, u32) {
    let stage_bytes = |sm: u32, sn: u32| -> usize {
        let elems_m = (t.0 * p.0 * sm) as usize;
        let elems_n = (t.1 * p.1 * sn) as usize;
        let elems_k = (t.2 * p.2) as usize;
        elems_m * elems_k * lhs_stage_size + elems_k * elems_n * rhs_stage_size
    };
    while stage_bytes(stage_m, stage_n) > max_shared {
        if stage_m > 1 && stage_m >= stage_n {
            stage_m /= 2;
        } else if stage_n > 1 {
            stage_n /= 2;
        } else if stage_m > 1 {
            stage_m /= 2;
        } else {
            break;
        }
    }
    (stage_m, stage_n)
}

/// Computes a [TilingBlueprint] depending on the problem kind
pub fn infer_blueprint_unit<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
    vector_sizes: &MatmulVectorSizes,
    options: UnitTilingBlueprintOptions,
    global_elems: &MatmulGlobalElems,
) -> (TilingBlueprint, MatmulElems) {
    let kind: MatmulKind = problem.into();
    let num_sms = client.properties().hardware.num_streaming_multiprocessors;
    // [daemon patch — see PATCH.md] the adapter budget the stage selection is clamped against.
    let max_shared = client.properties().hardware.max_shared_memory_size;
    let min_tile_size = usize::max(vector_sizes.lhs, vector_sizes.rhs);
    let min_tile_size = usize::max(vector_sizes.out, min_tile_size) as u32;
    let tile_size = u32::max(min_tile_size, 4);
    let dtypes = MatmulElems::from_globals(global_elems);

    let blueprint = match kind {
        MatmulKind::General => general_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            max_shared,
            options,
            &dtypes,
            vector_sizes,
        ),
        MatmulKind::MatVec => matvec_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            max_shared,
            options,
            &dtypes,
            vector_sizes,
        ),
        MatmulKind::VecMat => vecmat_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            max_shared,
            options,
            &dtypes,
            vector_sizes,
        ),
        MatmulKind::ScalarVec => scalarvec_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            max_shared,
            options,
            &dtypes,
            vector_sizes,
        ),
        MatmulKind::VecScalar => vecscalar_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            max_shared,
            options,
            &dtypes,
            vector_sizes,
        ),
        MatmulKind::InnerProduct => inner_product_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            max_shared,
            options,
            &dtypes,
            vector_sizes,
        ),
        MatmulKind::OuterProduct => outer_product_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            max_shared,
            options,
            &dtypes,
            vector_sizes,
        ),
        MatmulKind::ScalarProduct => scalar_product_unit_selector(
            problem,
            plane_dim,
            double_buffering,
            tile_size,
            num_sms,
            max_shared,
            options,
            &dtypes,
            vector_sizes,
        ),
    };

    (blueprint, dtypes)
}

/// (M, K) @ (K, N) → (M, N), with M, K, N > 1
#[allow(clippy::too_many_arguments)]
fn general_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    max_shared: usize,
    options: UnitTilingBlueprintOptions,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> TilingBlueprint {
    use cubek_std::MatrixLayout::*;

    // Manually tested for good performance on many shapes.
    let (tile_size, mut partition_size) =
        match (problem.lhs_layout, problem.rhs_layout, options.tile) {
            (RowMajor, _, TileSizeSelection::MinTileSize) => (
                (1, tile_size, tile_size),
                (
                    scale_partition(options.partition, problem.m, 4, 9),
                    2,
                    scale_partition(options.partition, problem.k, 2, 10),
                ),
            ),
            (ColMajor, RowMajor, TileSizeSelection::MinTileSize) => (
                (tile_size, tile_size, 1),
                (2, 2, scale_partition(options.partition, problem.k, 3, 10)),
            ),
            (ColMajor, ColMajor, _) | (_, _, TileSizeSelection::MaxTileSize) => (
                (tile_size, tile_size, tile_size),
                (
                    scale_partition(options.partition, problem.m, 2, 9),
                    2,
                    scale_partition(options.partition, problem.k, 2, 9),
                ),
            ),
        };

    let mut num_plane = 8;

    if double_buffering {
        if partition_size.0 > 2 {
            partition_size.0 /= 2;
        }
        if partition_size.2 > 2 {
            partition_size.2 /= 2;
        }
        num_plane /= 2;
    }

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::WithPlane {
            plane_dim,
            num_plane,
        },
        num_sms,
        GlobalOrder::SwizzleRow(4),
        options.stage,
        options.swizzle,
        max_shared,
        problem,
        dtypes,
        vector_sizes,
    )
}

/// (M, K) @ (K, 1) → (M, 1)
#[allow(clippy::too_many_arguments)]
fn matvec_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    max_shared: usize,
    options: UnitTilingBlueprintOptions,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> TilingBlueprint {
    let (tile_size, partition_size) = match (problem.lhs_layout, problem.rhs_layout) {
        (MatrixLayout::RowMajor, _) => ((1, 1, tile_size), (1, 1, tile_size * 2)),
        _ => ((tile_size, 1, tile_size), (1, 1, 1)),
    };

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed {
            m: (plane_dim / 2).max(1),
            n: 2,
        },
        num_sms,
        GlobalOrder::default(),
        StageScaling::Disabled,
        options.swizzle,
        max_shared,
        problem,
        dtypes,
        vector_sizes,
    )
}

/// (1, K) @ (K, N) → (1, N)
#[allow(clippy::too_many_arguments)]
fn vecmat_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    max_shared: usize,
    options: UnitTilingBlueprintOptions,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> TilingBlueprint {
    let (tile_size, partition_size) = ((1, tile_size, tile_size), (1, 1, 1));

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed {
            m: 2,
            n: (plane_dim / 2).max(1),
        },
        num_sms,
        GlobalOrder::default(),
        StageScaling::Disabled,
        options.swizzle,
        max_shared,
        problem,
        dtypes,
        vector_sizes,
    )
}

/// (1, 1) @ (1, N) → (1, N)
#[allow(clippy::too_many_arguments)]
fn scalarvec_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    max_shared: usize,
    options: UnitTilingBlueprintOptions,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> TilingBlueprint {
    use cubek_std::MatrixLayout::*;
    let (tile_size, partition_size) = match (problem.lhs_layout, problem.rhs_layout) {
        (RowMajor, RowMajor) => ((1, tile_size, tile_size), (1, 2, 1)),
        (RowMajor, ColMajor) => ((1, tile_size, tile_size), (1, 2, 1)),
        (ColMajor, RowMajor) => ((1, tile_size, tile_size), (1, 2, 1)),
        (ColMajor, ColMajor) => ((1, tile_size, tile_size), (2, 2, 1)),
    };

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed {
            m: 2,
            n: (plane_dim / 2).max(1),
        },
        num_sms,
        GlobalOrder::default(),
        StageScaling::Disabled,
        options.swizzle,
        max_shared,
        problem,
        dtypes,
        vector_sizes,
    )
}

/// (M, 1) @ (1, 1) → (M, 1)
#[allow(clippy::too_many_arguments)]
fn vecscalar_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    max_shared: usize,
    options: UnitTilingBlueprintOptions,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> TilingBlueprint {
    let (tile_size, partition_size) = ((tile_size, 1, 1), (1, 1, 1));

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed {
            m: (plane_dim / 2).max(1),
            n: 2,
        },
        num_sms,
        GlobalOrder::default(),
        StageScaling::Disabled,
        options.swizzle,
        max_shared,
        problem,
        dtypes,
        vector_sizes,
    )
}

/// (1, K) @ (K, 1) → (1, 1)
#[allow(clippy::too_many_arguments)]
fn inner_product_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    max_shared: usize,
    options: UnitTilingBlueprintOptions,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> TilingBlueprint {
    use cubek_std::MatrixLayout::*;
    let (tile_size, partition_size) = match (problem.lhs_layout, problem.rhs_layout) {
        (RowMajor, RowMajor) => ((1, 1, tile_size), (1, 1, 1)),
        (RowMajor, ColMajor) => ((1, 1, tile_size), (1, 1, 1)),
        (ColMajor, RowMajor) => ((1, 1, tile_size), (1, 1, 1)),
        (ColMajor, ColMajor) => ((1, 1, tile_size), (1, 1, 1)),
    };

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed { m: plane_dim, n: 1 }, // TODO: most planes does nothing.
        num_sms,
        GlobalOrder::default(),
        StageScaling::Disabled,
        options.swizzle,
        max_shared,
        problem,
        dtypes,
        vector_sizes,
    )
}

/// (M, 1) @ (1, N) → (M, N)
#[allow(clippy::too_many_arguments)]
fn outer_product_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    tile_size: u32,
    num_sms: Option<u32>,
    max_shared: usize,
    options: UnitTilingBlueprintOptions,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> TilingBlueprint {
    let (tile_size, partition_size) = ((tile_size, tile_size, 1), (1, 1, 1));

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::Fixed { m: 8, n: 8 },
        num_sms,
        GlobalOrder::default(),
        StageScaling::Disabled,
        options.swizzle,
        max_shared,
        problem,
        dtypes,
        vector_sizes,
    )
}

/// (1, 1) @ (1, 1) → (1, 1)
#[allow(clippy::too_many_arguments)]
fn scalar_product_unit_selector(
    problem: &MatmulProblem,
    plane_dim: u32,
    _double_buffering: bool,
    _tile_size: u32,
    num_sms: Option<u32>,
    max_shared: usize,
    options: UnitTilingBlueprintOptions,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> TilingBlueprint {
    let (tile_size, partition_size) = ((1, 1, 1), (1, 1, 1));

    selection(
        tile_size,
        partition_size,
        PartitionBuffering::Single,
        plane_dim,
        StageSelection::WithPlane {
            plane_dim,
            num_plane: 1,
        },
        num_sms,
        GlobalOrder::default(),
        StageScaling::Disabled,
        options.swizzle,
        max_shared,
        problem,
        dtypes,
        vector_sizes,
    )
}

enum StageSelection {
    WithPlane { plane_dim: u32, num_plane: u32 },
    Fixed { m: u32, n: u32 },
}

impl StageSelection {
    fn into_stages(self) -> (u32, u32) {
        match self {
            StageSelection::WithPlane {
                plane_dim: plane_size,
                num_plane: num_planes,
            } => {
                let num_units = num_planes * plane_size;
                closest_factor_pair(num_units)
            }
            StageSelection::Fixed { m, n } => (m.max(1), n.max(1)), // non-zero
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn selection(
    t: (u32, u32, u32),
    p: (u32, u32, u32),
    buffering: PartitionBuffering,
    plane_dim: u32,
    stage: StageSelection,
    num_sms: Option<u32>,
    global_order: GlobalOrder,
    stage_scaling: StageScaling,
    swizzle: bool,
    max_shared: usize,
    problem: &MatmulProblem,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> TilingBlueprint {
    let (stage_size_m, stage_size_n) = stage.into_stages();

    debug_assert!(
        stage_size_m > 0 && stage_size_n > 0,
        "Invalid stage size after normalization: m={stage_size_m}, n={stage_size_n}"
    );

    let (stage_size_m, stage_size_n) = match stage_scaling {
        StageScaling::Enabled(f) => (stage_size_m / f as u32, stage_size_n / f as u32),
        StageScaling::Disabled => (stage_size_m, stage_size_n),
    };

    // [daemon patch — see PATCH.md] Clamp the selection to the adapter's shared-memory budget
    // (tracel-ai/burn#4530, #4851): the launch validator would otherwise refuse the stage set,
    // and with autotune off there is no fallback.
    let (stage_size_m, stage_size_n) = clamp_stage_to_shared_memory(
        t,
        p,
        (stage_size_m, stage_size_n),
        (dtypes.lhs_stage.size(), dtypes.rhs_stage.size()),
        max_shared,
    );

    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(t.into())
        .with_partition_size(p.into())
        .with_stage_size((stage_size_m, stage_size_n, 1).into())
        .build()
        .unwrap();

    let cube_count_strategy = match num_sms {
        Some(num_sms) => CubeCountStrategy::Sm {
            num_sms,
            sm_usage: SmAllocation::Exact,
            cubes_first: false,
        },
        None => CubeCountStrategy::Flattened,
    };

    let hypercube = HypercubeBlueprint::builder()
        .global_order(global_order)
        .cube_count_strategy(cube_count_strategy)
        .build();

    let mut builder =
        TilingBlueprint::builder(TileMatmulKind::Register, tiling_scheme, plane_dim, problem)
            .partition_buffering(buffering)
            .hypercube_blueprint(hypercube);

    if swizzle {
        let lhs_swizzle_dim = match problem.lhs_layout {
            MatrixLayout::RowMajor => tiling_scheme.elements_per_stage_along_k() as usize,
            MatrixLayout::ColMajor => tiling_scheme.elements_per_stage_along_m() as usize,
        };
        let rhs_swizzle_dim = match problem.rhs_layout {
            MatrixLayout::RowMajor => tiling_scheme.elements_per_stage_along_n() as usize,
            MatrixLayout::ColMajor => tiling_scheme.elements_per_stage_along_k() as usize,
        };

        builder = builder.shared_swizzle(SwizzleModes {
            lhs: select_swizzle(lhs_swizzle_dim, dtypes.lhs_stage, vector_sizes.lhs),
            rhs: select_swizzle(rhs_swizzle_dim, dtypes.rhs_stage, vector_sizes.rhs),
            ..Default::default()
        })
    }

    builder.build()
}

/// All modes currently use atom size 16
const SWIZZLE_ATOM: usize = 16;

fn select_swizzle(swizzle_dim: usize, elem: StorageType, vector_size: VectorSize) -> SwizzleMode {
    // Can't swizzle if vector size > swizzle atom
    if elem.size() * vector_size > SWIZZLE_ATOM {
        return SwizzleMode::None;
    }
    let swizzle_dim_bytes = swizzle_dim * elem.size();
    if !swizzle_dim_bytes.is_power_of_two() {
        return SwizzleMode::None;
    }
    match swizzle_dim_bytes {
        32 => SwizzleMode::B32,
        64 => SwizzleMode::B64,
        _ => SwizzleMode::B128,
    }
}

/// Returns the factor pair `(a, b)` of `n` minimizing their difference,
/// with `a >= b` and `a * b == n`.
pub fn closest_factor_pair(n: u32) -> (u32, u32) {
    let sqrt_n = (n as f64).sqrt() as u32;
    for a in (1..=sqrt_n).rev() {
        if n.is_multiple_of(a) {
            return (n / a, a);
        }
    }
    (n, 1)
}

fn scale_partition(setting: PartitionScaling, axis: usize, max_exp: u32, div_exp: u32) -> u32 {
    if let PartitionScaling::Disabled = setting {
        return 2u32.pow(max_exp);
    }

    let exp = u32::min((axis as u32 / 2u32.pow(div_exp)) + 1, max_exp);
    2u32.pow(exp)
}

// [daemon patch — see PATCH.md]
#[cfg(test)]
mod daemon_patch_tests {
    use super::clamp_stage_to_shared_memory;

    /// The recorded Apple failure shape (tracel-ai/burn#4530): tile (4,4,4), partitions (4,2,4),
    /// stage (32,16) at f32 stages selects 32,768 B (lhs) + 8,192 B (rhs) = 40,960 B against the
    /// 32,768 B device limit. One halving of the m stage fits it.
    #[test]
    fn the_apple_overrun_clamps_to_the_device_limit() {
        assert_eq!(
            clamp_stage_to_shared_memory((4, 4, 4), (4, 2, 4), (32, 16), (4, 4), 32_768),
            (16, 16)
        );
    }

    /// The same selection under a 65,536 B limit (the green RADV lane) is untouched: the clamp
    /// changes nothing on adapters where the stage already fits.
    #[test]
    fn a_fitting_selection_is_untouched() {
        assert_eq!(
            clamp_stage_to_shared_memory((4, 4, 4), (4, 2, 4), (32, 16), (4, 4), 65_536),
            (32, 16)
        );
    }

    /// A pathological budget cannot loop forever: the walk floors at (1, 1) and leaves the final
    /// refusal to the launch validator.
    #[test]
    fn the_walk_stops_at_a_single_tile_pair() {
        assert_eq!(
            clamp_stage_to_shared_memory((4, 4, 4), (4, 2, 4), (32, 16), (4, 4), 1),
            (1, 1)
        );
    }
}
