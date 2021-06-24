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

//! Common utilities.

use itertools::FoldWhile::{self, Continue, Done};

/// Literals for boxed slices. Equivalent to `vec![...].into_boxed_slice()`.
///
/// ```
/// # use rowantlr::r#box;
/// assert_eq!(r#box![1, 2, 3], vec![1, 2, 3].into_boxed_slice());
/// assert_eq!(r#box![1, 2, 3, ], vec![1, 2, 3, ].into_boxed_slice());
/// assert_eq!(r#box![42; 3], vec![42; 3].into_boxed_slice());
/// ```
#[macro_export]
macro_rules! r#box {
    ($($es: expr),* $(,)?) => {
        ::std::vec![$($es),+].into_boxed_slice()
    };
    ($e: expr; $n: expr) => {
        ::std::vec![$e; $n].into_boxed_slice()
    }
}

/// Continue folding if the condition holds.
///
/// ```
/// # use rowantlr::utils::continue_if;
/// use itertools::FoldWhile::{Continue, Done};
/// assert_eq!(continue_if(true), Continue(()));
/// assert_eq!(continue_if(false), Done(()));
/// ```
pub fn continue_if(cond: bool) -> FoldWhile<()> {
    continue_if_with(cond, ())
}

/// Continue folding if the condition holds.
///
/// ```
/// # use rowantlr::utils::continue_if_with;
/// use itertools::FoldWhile::{Continue, Done};
/// assert_eq!(continue_if_with(true, 42), Continue(42));
/// assert_eq!(continue_if_with(false, 1), Done(1));
/// ```
pub fn continue_if_with<A>(cond: bool, a: A) -> FoldWhile<A> {
    if cond {
        Continue(a)
    } else {
        Done(a)
    }
}

/// Continue folding if the condition holds.
///
/// ```
/// # use rowantlr::utils::continue_unless;
/// use itertools::FoldWhile::{Continue, Done};
/// assert_eq!(continue_unless(true), Done(()));
/// assert_eq!(continue_unless(false), Continue(()));
/// ```
pub fn continue_unless(cond: bool) -> FoldWhile<()> {
    continue_unless_with(cond, ())
}

/// Continue folding if the condition does not hold.
///
/// ```
/// # use rowantlr::utils::continue_unless_with;
/// use itertools::FoldWhile::{Continue, Done};
/// assert_eq!(continue_unless_with(true, 42), Done(42));
/// assert_eq!(continue_unless_with(false, 1), Continue(1));
/// ```
pub fn continue_unless_with<A>(cond: bool, a: A) -> FoldWhile<A> {
    continue_if_with(!cond, a)
}
