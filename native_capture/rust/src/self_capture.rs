//! Self-capture exclusion: prevents the launcher window (e.g. the Terminal
//! window where `cyberalfred-capture start` was typed) and its spawned dialogs
//! from becoming learner evidence. Mirrors `CaptureManager.isLauncherWindow`
//! in the Swift helper.
//!
//! The core decision logic is pure and generic over the identity type `T`,
//! decoupled from live AXUIElement refs via `isEqual`/`parent` closures — so
//! it's fully unit-testable with plain values. The live AX wiring (recording
//! the launcher identity at start, passing CFEqual as `isEqual`) lives in
//! `main.rs` / `accessibility.rs`.

/// True if `window_identity` is the launcher window itself, or a dialog/sheet
/// spawned from it (e.g. a browser JS alert whose title becomes generic like
/// "localhost:8080 says"). Matches primarily by identity equality and AX
/// ancestry (stable even when title text changes), falling back to the
/// (bundle_id, title) pair only if neither identity check is available.
///
/// Generic over `T` (the identity type) so it's testable with plain integers
/// or strings — no live AXUIElement needed. `max_ancestor_depth` bounds the
/// walk up the parent chain (default 3, matching the Swift helper).
pub fn is_launcher_window<T: Clone>(
    window_bundle_id: Option<&str>,
    window_title: Option<&str>,
    window_identity: Option<&T>,
    launcher_bundle_id: Option<&str>,
    launcher_title: Option<&str>,
    launcher_identity: Option<&T>,
    is_equal: impl Fn(&T, &T) -> bool,
    parent: impl Fn(&T) -> Option<T>,
    max_ancestor_depth: usize,
) -> bool {
    if let (Some(launcher_id), Some(window_id)) = (launcher_identity, window_identity) {
        if is_equal(window_id, launcher_id) {
            return true;
        }
        if is_descendant_of(window_id, launcher_id, max_ancestor_depth, &parent, &is_equal) {
            return true;
        }
    }
    is_launcher_window_by_title(
        window_bundle_id,
        window_title,
        launcher_bundle_id,
        launcher_title,
    )
}

/// Bounded walk up an ancestor chain (e.g. AX parent), used to recognize a
/// dialog/sheet as belonging to a specific window without matching on title
/// text. Walks at most `max_depth` levels up.
pub fn is_descendant_of<T: Clone>(
    element: &T,
    ancestor: &T,
    max_depth: usize,
    parent: &impl Fn(&T) -> Option<T>,
    is_equal: &impl Fn(&T, &T) -> bool,
) -> bool {
    let mut current = element.clone();
    for _ in 0..max_depth {
        match parent(&current) {
            Some(next) => {
                if is_equal(&next, ancestor) {
                    return true;
                }
                current = next;
            }
            None => return false,
        }
    }
    false
}

/// Secondary/fallback comparison by (bundle_id, title) alone — kept pure so
/// it's unit-testable without a live Accessibility tree.
pub fn is_launcher_window_by_title(
    window_bundle_id: Option<&str>,
    window_title: Option<&str>,
    launcher_bundle_id: Option<&str>,
    launcher_title: Option<&str>,
) -> bool {
    let Some(launcher_bid) = launcher_bundle_id else {
        return false;
    };
    window_bundle_id == Some(launcher_bid) && window_title == launcher_title
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // -- is_launcher_window_by_title tests --

    #[test]
    fn by_title_matches_same_bundle_and_title() {
        assert!(is_launcher_window_by_title(
            Some("com.apple.Terminal"),
            Some("zsh — mentor"),
            Some("com.apple.Terminal"),
            Some("zsh — mentor"),
        ));
    }

    #[test]
    fn by_title_rejects_different_bundle() {
        assert!(!is_launcher_window_by_title(
            Some("com.google.Chrome"),
            Some("zsh — mentor"),
            Some("com.apple.Terminal"),
            Some("zsh — mentor"),
        ));
    }

    #[test]
    fn by_title_rejects_different_title() {
        assert!(!is_launcher_window_by_title(
            Some("com.apple.Terminal"),
            Some("vim file.txt"),
            Some("com.apple.Terminal"),
            Some("zsh — mentor"),
        ));
    }

    #[test]
    fn by_title_rejects_when_no_launcher_bundle() {
        assert!(!is_launcher_window_by_title(
            Some("com.apple.Terminal"),
            Some("zsh"),
            None,
            None,
        ));
    }

    // -- is_descendant_of tests --

    #[test]
    fn descendant_found_within_depth() {
        // tree: 1 -> 2 -> 3 -> 4  (parent direction: 4's parent is 3, etc.)
        let parents: HashMap<i32, i32> = [(4, 3), (3, 2), (2, 1), (1, 0)]
            .iter()
            .copied()
            .collect();
        let parent_fn = |e: &i32| parents.get(e).copied();
        assert!(is_descendant_of(&4, &1, 3, &parent_fn, &|a, b| a == b));
    }

    #[test]
    fn descendant_not_found_exceeds_depth() {
        let parents: HashMap<i32, i32> = [(4, 3), (3, 2), (2, 1), (1, 0)]
            .iter()
            .copied()
            .collect();
        let parent_fn = |e: &i32| parents.get(e).copied();
        // 4 -> 3 -> 2 -> 1 : depth 3 finds ancestor 1
        assert!(is_descendant_of(&4, &1, 3, &parent_fn, &|a, b| a == b));
        // depth 2 only gets to 2, doesn't find 1
        assert!(!is_descendant_of(&4, &1, 2, &parent_fn, &|a, b| a == b));
    }

    #[test]
    fn descendant_not_found_no_path() {
        let parents: HashMap<i32, i32> = [(4, 3), (3, 2)].iter().copied().collect();
        let parent_fn = |e: &i32| parents.get(e).copied();
        assert!(!is_descendant_of(&4, &1, 5, &parent_fn, &|a, b| a == b));
    }

    // -- is_launcher_window (full) tests --

    #[test]
    fn identity_match_excludes() {
        assert!(is_launcher_window(
            Some("com.apple.Terminal"),
            Some("different title now"),
            Some(&42i32),
            Some("com.apple.Terminal"),
            Some("zsh — mentor"),
            Some(&42i32),
            |a, b| a == b,
            |_| None,
            3,
        ));
    }

    #[test]
    fn ancestor_match_excludes_spawned_dialog() {
        // Dialog identity 99, parent chain 99 -> 42 (the launcher)
        assert!(is_launcher_window(
            Some("com.apple.Terminal"),
            Some("localhost:8080 says"),  // generic dialog title
            Some(&99i32),
            Some("com.apple.Terminal"),
            Some("zsh — mentor"),
            Some(&42i32),
            |a, b| a == b,
            |e| if *e == 99 { Some(42) } else { None },
            3,
        ));
    }

    #[test]
    fn no_identity_falls_back_to_title() {
        assert!(is_launcher_window(
            Some("com.apple.Terminal"),
            Some("zsh — mentor"),
            None, // no window identity available
            Some("com.apple.Terminal"),
            Some("zsh — mentor"),
            None, // no launcher identity available
            |a: &i32, b: &i32| a == b,
            |_| None,
            3,
        ));
    }

    #[test]
    fn unrelated_window_not_excluded() {
        assert!(!is_launcher_window(
            Some("com.google.Chrome"),
            Some("Google"),
            Some(&100i32),
            Some("com.apple.Terminal"),
            Some("zsh — mentor"),
            Some(&42i32),
            |a, b| a == b,
            |e| if *e == 100 { Some(50) } else { None },
            3,
        ));
    }

    #[test]
    fn same_bundle_different_window_not_excluded() {
        // Another Terminal window (different identity, different title) is NOT
        // the launcher window — only this one specific window is excluded.
        assert!(!is_launcher_window(
            Some("com.apple.Terminal"),
            Some("vim file.txt"),
            Some(&200i32),
            Some("com.apple.Terminal"),
            Some("zsh — mentor"),
            Some(&42i32),
            |a, b| a == b,
            |e| if *e == 200 { Some(201) } else { None },
            3,
        ));
    }
}
