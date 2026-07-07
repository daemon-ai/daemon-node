// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Small, SDK-free projections/normalisations between WhatsApp identifiers and the daemon model.
//!
//! These are the pure helpers the inbound path and the verb bodies share (JID classification,
//! Cloud-API recipient normalisation, addressing). They intentionally hold no SDK type so they are
//! unit-testable without a live client.

/// Normalise a Cloud API recipient to the digits Meta expects (country code, no `+`, spaces, or
/// dashes). A bare JID (`<number>@s.whatsapp.net`) is reduced to its user part.
pub fn bot_recipient(to: &str) -> String {
    let base = to.split('@').next().unwrap_or(to);
    base.chars().filter(|c| c.is_ascii_digit()).collect()
}

/// Classify whether an inbound message is *addressed* (may open/steer a turn) vs. ambient context.
///
/// A 1:1 (non-group) message is always addressed. In a group, when `mention_gating` is on only an
/// explicit `!command` turns the agent (WhatsApp Web mentions are opaque `@number` tokens that need
/// the account's own number to resolve — deferred); with gating off every group message is addressed.
pub fn classify_addressed(text: &str, is_group: bool, mention_gating: bool) -> bool {
    if !is_group {
        return true;
    }
    if !mention_gating {
        return true;
    }
    text.trim_start().starts_with('!')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipient_normalisation() {
        assert_eq!(bot_recipient("+1 (555) 123-4567"), "15551234567");
        assert_eq!(bot_recipient("15551234567@s.whatsapp.net"), "15551234567");
        assert_eq!(bot_recipient("15551234567"), "15551234567");
    }

    #[test]
    fn addressing_dm_always_on() {
        assert!(classify_addressed("hi", false, true));
        assert!(classify_addressed("hi", false, false));
    }

    #[test]
    fn addressing_group_gated_on_command() {
        assert!(classify_addressed("!help", true, true));
        assert!(!classify_addressed("just chatting", true, true));
        // gating off => every group message is addressed.
        assert!(classify_addressed("just chatting", true, false));
    }
}
