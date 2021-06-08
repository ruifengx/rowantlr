/*
 * rowantlr: ANTLR-like parser generator framework targetting rowan
 * Copyright (C) 2021  Xie Ruifeng
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

//! `LL(1)` backend for rowantlr, and some common utilities.
//!
//! For each non-terminal `N`:
//! - `DEDUCE_TO_EMPTY(N)` indicates whether `N —→* ε`.
//! - `FIRST(N)` collects all terminal `a` such that `N —→* a β` (`β ∈ V*`).
//!
//! Below is an illustration, one can:
//! - use the [`Grammar`] API to create new grammars.
//! - use [`calc_deduce_to_empty`] to calculate `DEDUCE_TO_EMPTY`.
//! - use [`calc_first`] to calculate `FIRST`.
//!
//! ```
//! #![allow(non_snake_case)]
//! use rowantlr::r#box;
//! use rowantlr::ir::grammar::{Grammar, epsilon, Symbol::*};
//! use rowantlr::backend::ll1::{calc_deduce_to_empty, calc_first};
//!
//! let mut g = Grammar::<&'static str>::new();
//! let [E, T, E_, T_, F] = g.add_non_terminals();
//! // E  —→ T E'
//! g.mark_as_start(E);
//! g.add_rule(E, vec![NonTerminal(T), NonTerminal(E_)]);
//! // E' —→ + T E' | ε
//! g.add_rule(E_, vec![Terminal("+"), NonTerminal(T), NonTerminal(E_)]);
//! g.add_rule(E_, epsilon());
//! // T  —→ F T'
//! g.add_rule(T, vec![NonTerminal(F), NonTerminal(T_)]);
//! // T' —→ * F T' | ε
//! g.add_rule(T_, vec![Terminal("*"), NonTerminal(F), NonTerminal(T_)]);
//! g.add_rule(T_, epsilon());
//! // F  —→ ( E ) | id
//! g.add_rule(F, vec![Terminal("("), NonTerminal(E), Terminal(")")]);
//! g.add_rule(F, vec![Terminal("id")]);
//!
//! // Calculate `DEDUCE_TO_EMPTY` and `FIRST`.
//! let deduce_to_empty = calc_deduce_to_empty(&g);
//! assert_eq!(deduce_to_empty, r#box![false, false, false, true, true, false]);
//! let first = calc_first(&g, &deduce_to_empty);
//! assert_eq!(first, r#box![
//!     r#box!["(", "id"],
//!     r#box!["(", "id"],
//!     r#box!["(", "id"],
//!     r#box!["+"],
//!     r#box!["*"],
//!     r#box!["(", "id"],
//! ]);
//! ```

use std::collections::BTreeSet;
use itertools::Itertools;
use itertools::FoldWhile::*;

use crate::ir::grammar::{Grammar, Symbol};
use crate::utils::continue_if_with;

/// Calculate the `DEDUCE_TO_EMPTY` set for each non-terminal.
pub fn calc_deduce_to_empty<A>(g: &Grammar<A>) -> Box<[bool]> {
    let mut res = vec![false; g.rules.len()];
    let mut updated = true;
    while updated {
        updated = false;
        for (nt, rules) in g.rules.iter().enumerate() {
            let new_val = rules.iter().any(|expr| {
                !expr.iter().fold_while((), |(), x| match x {
                    Symbol::Terminal(_) => Done(()),
                    Symbol::NonTerminal(nt) =>
                        continue_if_with(res[nt.get()], ()),
                }).is_done()
            });
            updated |= res[nt] != new_val;
            res[nt] = new_val;
        }
    }
    res.into_boxed_slice()
}

/// Calculate the `FIRST` set for each non-terminal.
pub fn calc_first<A: Ord + Clone>(g: &Grammar<A>, deduce_to_empty: &[bool]) -> Box<[Box<[A]>]> {
    let mut res = vec![BTreeSet::new(); g.rules.len()];
    let mut updated = true;
    let mut count = 0;
    while updated {
        updated = false;
        for (nt, expr) in g.rules_iter() {
            res[nt] = expr.iter().fold_while(
                std::mem::take(&mut res[nt]),
                |mut cur, x| match x {
                    Symbol::Terminal(t) => {
                        updated |= cur.insert(t.clone());
                        Done(cur)
                    }
                    Symbol::NonTerminal(nt) => {
                        let nt = nt.get();
                        for a in &res[nt] {
                            updated |= cur.insert(a.clone());
                        }
                        continue_if_with(deduce_to_empty[nt], cur)
                    }
                }).into_inner();
        }
        count += 1;
        if count > 10 { panic!() }
    }
    res.into_iter().map(|s| s.into_iter().collect()).collect()
}
