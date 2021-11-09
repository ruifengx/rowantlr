/*
 * rowantlr: ANTLR-like parser generator framework targeting rowan
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

//! Tuple-related utilities.

/// Get a shared reference for each element of a tuple.
///
/// # Examples
///
/// ```
/// # use rowantlr::utils::tuple::{TupleBorrow, TupleCompare};
/// let data: (String, Vec<&'static str>, usize) =
///     ("answer".to_string(), vec!["life", "the universe", "everything"], 42);
/// let borrowed: (&String, &Vec<&'static str>, &usize) = data.tuple_borrow();
/// ```
pub trait TupleBorrow<'a> {
    /// A tuple type shaped as `Self`, with each element replaced with its shared reference.
    type Borrowed;
    /// Borrow each element of `self`.
    fn tuple_borrow(&'a self) -> Self::Borrowed;
}

macro_rules! invoke_macro {
    ($mac: ident ($t: ident $(, $ts: ident)*) ($i: tt $(, $idx: tt)*)) => {
        invoke_macro!(step $mac ($t,) ($($ts),*) ($i,) ($($idx),*));
    };
    (step $mac: ident ($($ts0: ident, )+) () ($($idx0: tt, )+) ()) => {
        $mac!(($($ts0),*) ($($idx0),*));
    };
    (step $mac: ident ($($ts0: ident, )+) ($t: ident $(, $ts: ident)*)
                      ($($idx0: tt, )+) ($i: tt $(, $idx: tt)*)) => {
        $mac!(($($ts0),*) ($($idx0),*));
        invoke_macro!(step $mac ($($ts0, )+ $t,) ($($ts),*) ($($idx0, )+ $i,) ($($idx),*));
    }
}

macro_rules! tuple_borrow {
    (($($ts: ident),+) ($($idx: tt),+)) => {
        impl<'a, $($ts: 'a),+> TupleBorrow<'a> for ($($ts,)+) {
            type Borrowed = ($(&'a $ts,)+);
            fn tuple_borrow(&'a self) -> Self::Borrowed {
                ($(&self.$idx,)+)
            }
        }
    }
}

invoke_macro! {
    tuple_borrow
    (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11)
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
}

/// Two tuple type, `Self` and `Other`, are comparable.
///
/// Typically `Other` is a "prefix" of `Self` (modulo [`Borrow`](std::borrow::Borrow)), as is the
/// case for `Self = (A, B, C)` and `Other = (X, Y)`, where `A: Borrow<X>` and `B: Borrow<Y>`.
///
/// # Examples
///
/// ```
/// # use std::cmp::Ordering;
/// # use rowantlr::utils::tuple::{TupleBorrow, TupleCompare};
/// let data: (String, Vec<&'static str>, usize) =
///     ("answer".to_string(), vec!["life", "the universe", "everything"], 42);
/// assert_eq!(data.tuple_compare(&("answer", )), Ordering::Equal);
/// assert_eq!(data.tuple_compare(&("question", )), Ordering::Less);
/// assert_eq!(data.tuple_compare(&("answer", &["life"] as &[_])), Ordering::Greater);
/// assert_eq!(data.tuple_compare(&data.tuple_borrow()), Ordering::Equal);
/// ```
pub trait TupleCompare<Other: ?Sized> {
    /// Compare two tuples.
    fn tuple_compare(&self, other: &Other) -> std::cmp::Ordering;
}

/// Drop the `Init` part of the tuple `Self`, and borrow the `Rest` part element-wise.
pub trait TupleRest<'a, Init: ?Sized> {
    /// References of tuple `Self` with `Init` part removed.
    type Rest;
    /// Borrow the `Rest` part element-wise.
    fn borrow_rest(&'a self) -> Self::Rest;
}

/// Split the tuple `Self` into two tuples: `Init` and `Tail`.
pub trait TupleSplit<const N: usize> {
    /// Initial part of `Self`.
    type Init;
    /// Trailing part of `Self`.
    type Tail;
    /// Split the tuple.
    fn tuple_split(self) -> (Self::Init, Self::Tail);
}

/// Reverse operation for [`TupleSplit`].
pub trait TupleMerge<Init, Tail> {
    /// Merge back from the two parts.
    fn tuple_merge(init: Init, tail: Tail) -> Self;
}

/// Rotation of tuples: e.g. (N = 1) `(a, b, c)` ⇒ `(c, a, b)`.
/// Done by a [`TupleSplit`] followed by a [`TupleMerge`].
pub trait TupleRotate<const N: usize>: TupleSplit<N> {
    /// Rotated version of `Self`.
    type Rotated: TupleMerge<Self::Tail, Self::Init>;
    /// Swap the initial and trailing part of `Self`.
    #[inline(always)]
    fn tuple_rotate(self) -> Self::Rotated where Self: Sized {
        let (init, tail) = self.tuple_split();
        Self::Rotated::tuple_merge(tail, init)
    }
}

macro_rules! scan_macro {
    ($mac: ident ($u: ident $(, $us: ident)+)
                 ($t: ident $(, $ts: ident)+)
                 ($i: tt $(, $idx: tt)+)) => {
        scan_macro! {accum $mac
            ($u, ) ($t, ) ($i, )
            ($($us, )+) ($($ts, )+) ($($idx, )+)
        }
    };
    (accum $mac: ident ($($us: ident, )+) ($($ts: ident, )+) ($($idx: tt, )+) () () ()) => {
        scan_macro! {split $mac () () () ($($us, )+) ($($ts, )+) ($($idx, )+) (0)}
    };
    (accum $mac: ident ($($us: ident, )+) ($($ts: ident, )+) ($($idx: tt, )+)
                       ($u0: ident, $($us0: ident, )*)
                       ($t0: ident, $($ts0: ident, )*)
                       ($i0: tt, $($idx0: tt, )*)) => {
        scan_macro! {split $mac () () () ($($us, )+) ($($ts, )+) ($($idx, )+) (0)}
        scan_macro! {accum $mac
            ($($us, )+ $u0, ) ($($ts, )+ $t0, ) ($($idx, )+ $i0, )
            ($($us0, )*) ($($ts0, )*) ($($idx0, )*)
        }
    };
    (split $mac: ident ($($us: ident, )*) ($($ts: ident, )*) ($($idx: tt, )*) () () () ($n: expr)) => {
        $mac! {($n) ($($us),*) ($($ts),*) ($($idx),*) () () ()}
    };
    (split $mac: ident ($($us: ident, )*) ($($ts: ident, )*) ($($idx: tt, )*)
                       ($u0: ident, $($us0: ident, )*)
                       ($t0: ident, $($ts0: ident, )*)
                       ($i0: tt, $($idx0: tt, )*)
                       ($n: expr)) => {
        $mac! {
            ($n) ($($us),*) ($($ts),*) ($($idx),*)
            ($u0 $(, $us0)*) ($t0 $(, $ts0)*) ($i0 $(, $idx0)*)
        }
        scan_macro! {split $mac
            ($($us, )* $u0, ) ($($ts, )* $t0, ) ($($idx, )* $i0, )
            ($($us0, )*) ($($ts0, )*) ($($idx0, )*) ($n + 1)
        }
    }
}

macro_rules! tuple_compare {
    (($n: expr) ($u: ident) ($t: ident) ($idx: tt) () () ()) => {};
    (($n: expr) () () () ($uR: ident) ($tR: ident) ($idxR: tt)) => {};
    (($n: expr) ($($us: ident),*) ($($ts: ident),*) ($($idx: tt),*)
    ($($usR: ident),*) ($($tsR: ident),*) ($($idxR: tt),*)) => {
        impl<'a, $($us, )* $($usR, )* $($ts: ?Sized, )*> TupleCompare<($(&'a $ts, )*)>
            for ($($us, )* $($usR, )*)
            where $($us: ::std::borrow::Borrow<$ts>, )*
                  $($ts: ::std::cmp::Ord, )* {
            fn tuple_compare(&self, _other: &($(&'a $ts, )*)) -> ::std::cmp::Ordering {
                ::std::cmp::Ordering::Equal
                $(.then_with(|| self.$idx.borrow().cmp(_other.$idx)))*
            }
        }
        impl<'a, $($us, )* $($usR: 'a, )* $($ts, )*>
            TupleRest<'a, ($($ts, )*)> for ($($us, )* $($usR, )*) {
            #[allow(unused_parens)]
            type Rest = ($(&'a $usR),*);
            #[allow(clippy::unused_unit)]
            #[inline(always)]
            fn borrow_rest(&'a self) -> Self::Rest {
                ($(&self.$idxR),*)
            }
        }
        impl<$($us, )* $($usR, )*> TupleSplit<{$n}> for ($($us, )* $($usR, )*) {
            #[allow(unused_parens)]
            type Init = ($($us),*);
            #[allow(unused_parens)]
            type Tail = ($($usR),*);
            #[inline(always)]
            fn tuple_split(self) -> (Self::Init, Self::Tail) {
                (($(self.$idx),*), ($(self.$idxR),*))
            }
        }
        #[allow(unused_parens)]
        impl<$($us, )* $($usR, )*> TupleMerge<($($us),*), ($($usR),*)> for ($($us, )* $($usR, )*) {
            #[inline(always)]
            fn tuple_merge(init: ($($us),*), tail: ($($usR),*)) -> Self {
                #[allow(non_snake_case)]
                let ($($us),*) = init;
                #[allow(non_snake_case)]
                let ($($usR),*) = tail;
                ($($us, )* $($usR, )*)
            }
        }
        impl<$($us, )* $($usR, )*> TupleRotate<{$n}> for ($($us, )* $($usR, )*) {
            type Rotated = ($($usR, )* $($us, )*);
        }
    }
}

scan_macro! {
    tuple_compare
    (U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11)
    (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11)
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
}
