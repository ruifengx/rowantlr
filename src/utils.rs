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

/// Literals for [`std::num::NonZeroUsize`]. Avoids runtime checks.
///
/// ```
/// # use rowantlr::nonzero;
/// use std::num::NonZeroUsize;
/// assert_eq!(nonzero!(1), NonZeroUsize::new(1).unwrap());
/// ```
#[macro_export]
macro_rules! nonzero {
    ($n: expr) => {{
        const __nonzero_n : usize = $n;
        let _ = [(); (__nonzero_n.count_ones() as usize) - 1];
        unsafe { ::std::num::NonZeroUsize::new_unchecked(__nonzero_n) }
    }}
}
