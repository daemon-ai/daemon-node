//! A bounded, monotonically-cursored ring: the shared core of the daemon's cursored live streams
//! (the gold-standard cursored contract, `daemon-event-io-spec` §5.4.1). Each pushed item is
//! assigned a monotonic id (seq/cursor) starting at 1; an overflow eviction raises a `floor`, so a
//! reader whose cursor fell below it knows it lagged and must resync. Pure + sync: the
//! broadcast/subscribe machinery and the lag -> `Reset`/`ResyncNeeded` mapping stay in the
//! per-transport call sites (`MergedLog`, `NodeEventFeed`, `WorkspaceFs` watch).

use std::collections::VecDeque;

/// An item delivered from a cursored stream, generalizing the hand-rolled `LogStreamItem`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CursoredItem<T> {
    /// The next retained element, in id order.
    Item(T),
    /// The reader fell behind the retained window and must resync from `head`.
    Lagged {
        /// The current live edge (highest id assigned).
        head: u64,
    },
}

/// A bounded ring of `(id, T)` with a monotonic id and a retained-window `floor`.
///
/// `cap == 0` means unbounded (never evicts, so it never reports `lagged` from `page`) - used by the
/// merged session log, whose retention is the full history. A non-zero `cap` evicts the oldest
/// entries past the cap and raises `floor` past them.
#[derive(Clone, Debug)]
pub struct CursoredRing<T> {
    ring: VecDeque<(u64, T)>,
    next_id: u64, // id to assign on the next push (1 before the first push)
    floor: u64,   // highest id evicted on overflow; a reader at `after < floor` has lagged
    cap: usize,   // 0 == unbounded
}

impl<T> Default for CursoredRing<T> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<T> CursoredRing<T> {
    /// A ring retaining at most `cap` items (`0` == unbounded).
    pub fn new(cap: usize) -> Self {
        Self {
            ring: VecDeque::new(),
            next_id: 1,
            floor: 0,
            cap,
        }
    }

    /// Append `item`, returning its assigned id. On overflow the oldest entries are evicted and the
    /// `floor` is raised past them, so a reader behind the retained window will then see `lagged()`.
    pub fn push(&mut self, item: T) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.ring.push_back((id, item));
        if self.cap != 0 {
            while self.ring.len() > self.cap {
                if let Some((evicted, _)) = self.ring.pop_front() {
                    self.floor = self.floor.max(evicted);
                }
            }
        }
        id
    }

    /// Remove every retained item matching `pred` WITHOUT raising the floor - this is
    /// coalescing/superseding (e.g. dropping a stale per-session advance), not eviction, so a reader
    /// at an older cursor is not considered lagged by it. Returns the number removed.
    pub fn coalesce(&mut self, pred: impl Fn(&T) -> bool) -> usize {
        let before = self.ring.len();
        self.ring.retain(|(_, t)| !pred(t));
        before - self.ring.len()
    }

    /// The highest id assigned so far (the live edge); `0` before the first push.
    pub fn head(&self) -> u64 {
        self.next_id - 1
    }

    /// The resync floor (highest id evicted on overflow); `0` while nothing has been evicted.
    pub fn floor(&self) -> u64 {
        self.floor
    }

    /// Whether a reader at cursor `after` fell behind the retained window (its next-wanted entries
    /// were evicted). Always `false` for an unbounded ring.
    pub fn lagged(&self, after: u64) -> bool {
        after < self.floor
    }

    /// Number of retained entries.
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    /// Whether the ring currently retains no entries.
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Iterate retained `(id, item)` pairs in id order (for sites needing custom filtering).
    pub fn iter(&self) -> impl Iterator<Item = &(u64, T)> {
        self.ring.iter()
    }
}

impl<T: Clone> CursoredRing<T> {
    /// Retained `(id, item)` with id `> after`, in order, capped at `max` (`0` == no cap).
    pub fn page(&self, after: u64, max: usize) -> Vec<(u64, T)> {
        let mut out = Vec::new();
        for (id, item) in &self.ring {
            if *id > after {
                out.push((*id, item.clone()));
                if max != 0 && out.len() >= max {
                    break;
                }
            }
        }
        out
    }

    /// The backlog a subscriber at cursor `after` should receive before the live tail: a single
    /// `Lagged { head }` if it fell behind, otherwise the retained items past `after` as `Item`s.
    /// The caller appends the live broadcast stream after this.
    pub fn backlog(&self, after: u64) -> Vec<CursoredItem<T>> {
        if self.lagged(after) {
            return vec![CursoredItem::Lagged { head: self.head() }];
        }
        self.page(after, 0)
            .into_iter()
            .map(|(_, t)| CursoredItem::Item(t))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbounded_assigns_monotonic_ids_and_never_lags() {
        let mut r = CursoredRing::<&str>::new(0);
        assert_eq!(r.head(), 0);
        assert_eq!(r.push("a"), 1);
        assert_eq!(r.push("b"), 2);
        assert_eq!(r.push("c"), 3);
        assert_eq!(r.head(), 3);
        assert_eq!(r.floor(), 0);
        assert!(!r.lagged(0)); // nothing evicted -> a from-start reader is never behind
        assert_eq!(r.page(0, 0).len(), 3);
        assert_eq!(
            r.page(1, 0).iter().map(|(_, t)| *t).collect::<Vec<_>>(),
            vec!["b", "c"]
        );
    }

    #[test]
    fn overflow_evicts_oldest_and_raises_floor() {
        let mut r = CursoredRing::<&str>::new(2);
        r.push("a"); // id 1
        r.push("b"); // id 2
        r.push("c"); // id 3 -> evicts id 1
        assert_eq!(r.head(), 3);
        assert_eq!(r.floor(), 1);
        assert_eq!(r.len(), 2);
        assert!(r.lagged(0)); // wanted id 1.. which was evicted
        assert!(!r.lagged(1)); // wants id 2,3 -> both retained
        assert!(!r.lagged(2));
        assert_eq!(
            r.page(1, 0).iter().map(|(_, t)| *t).collect::<Vec<_>>(),
            vec!["b", "c"]
        );
        assert_eq!(
            r.page(2, 0).iter().map(|(_, t)| *t).collect::<Vec<_>>(),
            vec!["c"]
        );
    }

    #[test]
    fn page_respects_max() {
        let mut r = CursoredRing::<u32>::new(0);
        for n in 0..5 {
            r.push(n);
        }
        assert_eq!(r.page(0, 2).len(), 2);
        assert_eq!(r.page(0, 0).len(), 5);
    }

    #[test]
    fn coalesce_removes_without_raising_floor() {
        let mut r = CursoredRing::<&str>::new(0);
        r.push("a"); // 1
        r.push("b"); // 2
        r.push("a"); // 3
        let removed = r.coalesce(|t| *t == "a");
        assert_eq!(removed, 2);
        assert_eq!(r.floor(), 0); // coalescing is not eviction
        assert!(!r.lagged(0));
        assert_eq!(
            r.page(0, 0).iter().map(|(_, t)| *t).collect::<Vec<_>>(),
            vec!["b"]
        );
    }

    #[test]
    fn backlog_is_lagged_marker_or_items() {
        let mut r = CursoredRing::<&str>::new(2);
        r.push("a"); // 1
        r.push("b"); // 2
        r.push("c"); // 3 -> floor 1
        assert_eq!(r.backlog(0), vec![CursoredItem::Lagged { head: 3 }]);
        assert_eq!(r.backlog(2), vec![CursoredItem::Item("c")]);
    }
}
