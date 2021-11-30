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

//! Common utilities.

use std::any::type_name;
use std::iter::FromIterator;
use std::fmt::{Display, Formatter};
use std::ops::{Bound, RangeBounds};
use derivative::Derivative;
use tuple::{TupleBorrow, TupleCompare, TupleRest, TupleRotate, TupleSplit};

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

// Generally we do not expect anything to overflow `usize`, because we are usually on 64bit
// platforms, and `u128` is not going to appear very often. Under such circumstances, the check
// should be completely optimised away by the compiler.
//
// That said, signed integers may actually underflow `usize`.
macro_rules! idx {
    ($i: expr) => {{
        let __idx: usize = TryInto::try_into($i)
            .expect("using this integer as index under/overflows `usize`");
        __idx
    }}
}

macro_rules! narrow {
    ($n: expr => $t: ty) => {{
        let __narrow_input: usize = $n;
        match TryInto::<$t>::try_into(__narrow_input) {
            Ok(__narrow_result) => __narrow_result,
            Err(__narrow_error) => crate::utils::narrow_failed(
                stringify!($n), __narrow_input, __narrow_error,
            ),
        }
    }};
    ($n: expr) => {
        narrow!($n => _)
    }
}

pub mod tuple;
pub mod partition_refinement;
pub mod interval;

#[cold]
#[inline(never)]
#[track_caller]
pub(crate) fn narrow_failed<E: Display, T>(expr_text: &str, value: usize, error: E) -> T {
    narrow_failed_impl(expr_text, value, error, type_name::<T>())
}

#[inline(always)]
#[track_caller]
pub(crate) fn narrow_failed_impl<E: Display>(
    expr_text: &str, value: usize,
    error: E, target_type: &str) -> ! {
    panic!("narrowing '{}' (= {}) to {} failed: {}", expr_text, value, target_type, error)
}

/// Compare references as if they were raw pointers.
pub mod by_address {
    use std::cmp::Ordering;

    /// Fast-forward to [`PartialOrd`] for pointers. To be used by `Derivative`.
    #[inline(always)]
    pub fn eq<T: ?Sized>(x: &&T, y: &&T) -> bool {
        std::ptr::eq(*x, *y)
    }

    /// Fast-forward to [`PartialOrd`] for pointers. To be used by `Derivative`.
    #[inline(always)]
    pub fn partial_cmp<T: ?Sized>(x: &&T, y: &&T) -> Option<Ordering> {
        let x: *const T = *x;
        let y: *const T = *y;
        x.partial_cmp(&y)
    }

    /// Fast-forward to [`Ord`] for pointers. To be used by `Derivative`.
    #[inline(always)]
    pub fn cmp<T: ?Sized>(x: &&T, y: &&T) -> Ordering {
        let x: *const T = *x;
        let y: *const T = *y;
        x.cmp(&y)
    }
}

/// Wraps a [`DisplayDot2TeX`] type for convenient formatting.
pub struct Dot2TeX<'a, A: ?Sized, Env: ?Sized>(&'a A, &'a Env);

impl<'a, A: DisplayDot2TeX<Env> + ?Sized, Env: ?Sized> Display for Dot2TeX<'a, A, Env> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt_dot2tex(self.1, f)
    }
}

/// Display the automata by going through [`dot2tex`](https://github.com/kjellmf/dot2tex/)
/// via [`dot2texi`](https://www.ctan.org/pkg/dot2texi) and then `TikZ` in `LaTeX`.
///
/// To typeset this automata, make sure `dot2tex` is in `PATH`.
pub trait DisplayDot2TeX<Env: ?Sized = ()> {
    /// Formats in `dot2tex` to the given formatter.
    fn fmt_dot2tex(&self, env: &'_ Env, f: &mut Formatter<'_>) -> std::fmt::Result;
    /// Wraps the type for convenient formatting.
    fn display_dot2tex<'a>(&'a self, env: &'a Env) -> Dot2TeX<'a, Self, Env> {
        Dot2TeX(self, env)
    }
}

/// Simple versions of the types and traits.
pub mod simple {
    use std::fmt::Formatter;

    /// Provide short-cut methods for [`DisplayDot2TeX`] with a nullary environment.
    pub trait DisplayDot2TeX: super::DisplayDot2TeX + private::Sealed {
        /// Formats in `dot2tex` to the given formatter.
        fn fmt_dot2tex_(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            self.fmt_dot2tex(&(), f)
        }
        /// Wraps the type for convenient formatting.
        fn display_dot2tex_(&self) -> super::Dot2TeX<Self, ()> {
            self.display_dot2tex(&())
        }
    }

    impl<T: super::DisplayDot2TeX> DisplayDot2TeX for T {}

    mod private {
        pub trait Sealed {}

        impl<T> Sealed for T {}
    }
}

/// Declares a set of types that are safe to display as `dot2tex` format directly via [`Display`].
#[macro_export]
macro_rules! display_dot2tex_via_display {
    ($($t: ty),+ $(,)?) => {
        $(
            impl<Env: ?Sized> DisplayDot2TeX<Env> for $t {
                fn fmt_dot2tex(&self, _: &Env, f: &mut Formatter<'_>) -> std::fmt::Result {
                    write!(f, "{}", self)
                }
            }
        )+
    }
}

macro_rules! display_dot2tex_via_deref {
    ($t: ty $(where $($p: tt)+)?) => {
        impl<$($($p)+ ,)? Env: ?Sized> DisplayDot2TeX<Env> for $t {
            fn fmt_dot2tex(&self, env: &Env, f: &mut Formatter<'_>) -> std::fmt::Result {
                <<Self as ::std::ops::Deref>::Target as DisplayDot2TeX<Env>>::fmt_dot2tex(
                    ::std::ops::Deref::deref(self), env, f)
            }
        }
    }
}

display_dot2tex_via_display!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
display_dot2tex_via_deref!(String);
display_dot2tex_via_deref!(&'a T where 'a, T: DisplayDot2TeX<Env> + ?Sized);
display_dot2tex_via_deref!(Box<T> where T: DisplayDot2TeX<Env> + ?Sized);
display_dot2tex_via_deref!(std::rc::Rc<T> where T: DisplayDot2TeX<Env> + ?Sized);
display_dot2tex_via_deref!(std::sync::Arc<T> where T: DisplayDot2TeX<Env> + ?Sized);
display_dot2tex_via_deref!(std::borrow::Cow<'_, B> where B: DisplayDot2TeX<Env> + ToOwned + ?Sized);

const TEX_SPECIAL: &[char] = &['#', '$', '%', '&', '\\', '^', '_', '{', '}', '~'];

impl<Env: ?Sized> DisplayDot2TeX<Env> for str {
    fn fmt_dot2tex(&self, _: &Env, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut rest = self;
        while let Some(idx) = rest.find(TEX_SPECIAL) {
            if idx != 0 { f.write_str(&rest[..idx])?; }
            let mut rest_chars = rest[idx..].chars();
            f.write_str(match rest_chars.next().unwrap() {
                '#' => r"\#",
                '$' => r"\$",
                '%' => r"\%",
                '&' => r"\&",
                '\\' => r"\backslash ",
                '^' => r"\char94 ",
                '_' => r"\_",
                '{' => r"\{",
                '}' => r"\}",
                '~' => r"\textasciitilde ",
                _ => unreachable!(),
            })?;
            rest = rest_chars.as_str();
        }
        if !rest.is_empty() { f.write_str(rest)?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::simple::DisplayDot2TeX;

    #[test]
    #[should_panic(expected = "narrowing 'too_large' (= 300) to u8 failed")]
    fn test_narrow() {
        let too_large = 300_usize;
        let _ = narrow!(too_large => u8);
    }

    #[test]
    fn test_str_fmt_dot2tex() {
        assert_eq!(&format!("{}", r#"Magic${Hash'}^#_\~&%"#.display_dot2tex_()),
                   r#"Magic\$\{Hash'\}\char94 \#\_\backslash \textasciitilde \&\%"#);
    }
}

/// More utility functions for iterators.
pub trait IterHelper: Iterator {
    /// [`Iterator::map`] followed by [`Iterator::reduce`], with lifetime issues properly handled.
    fn reduce_map<V, A>(mut self, visitor: &mut V,
                        f: impl Fn(&mut V, Self::Item) -> A,
                        g: impl Fn(&mut V, A, A) -> A) -> Option<A>
        where Self: Sized, V: ?Sized {
        let first = f(visitor, self.next()?);
        Some(self.fold(first, |res, x| {
            let a = f(visitor, x);
            g(visitor, res, a)
        }))
    }
}

impl<I: Iterator> IterHelper for I {}

/// An immutable, compact, flat dictionary.
///
/// # Examples
///
/// `Dict`s can be created from arrays:
/// ```
/// # use rowantlr::utils::Dict;
/// let solar_distance = Dict::from([
///     ("Mercury",   5790_0000),
///     ("Venus",   1_0820_0000),
///     ("Earth",   1_4960_0000),
///     ("Mars",    2_2790_0000),
/// ]);
/// ```
/// from [`Vec`]s:
/// ```
/// # use rowantlr::utils::Dict;
/// let solar_distance = Dict::from(vec![("Mercury", 5790_0000), /* ... */]);
/// ```
/// from boxed slices:
/// ```
/// # use rowantlr::r#box;
/// # use rowantlr::utils::Dict;
/// let solar_distance = Dict::from(r#box![("Mercury", 5790_0000), /* ... */]);
/// ```
/// or from iterators:
/// ```
/// # use rowantlr::utils::Dict;
/// let solar_distance = [("Mercury", 5790_0000), /* ... */].into_iter().collect::<Dict<_>>();
/// ```
///
/// Items of a `Dict` is always sorted. Use [`Dict::into_raw`] to obtain the back buffer:
/// ```
/// # use rowantlr::r#box;
/// # use rowantlr::utils::Dict;
/// # let solar_distance = Dict::from([
/// #     ("Mercury",   5790_0000),
/// #     ("Venus",   1_0820_0000),
/// #     ("Earth",   1_4960_0000),
/// #     ("Mars",    2_2790_0000),
/// # ]);
/// assert_eq!(solar_distance.into_raw(), r#box![
///     ("Earth",   1_4960_0000),
///     ("Mars",    2_2790_0000),
///     ("Mercury",   5790_0000),
///     ("Venus",   1_0820_0000),
/// ]);
/// ```
///
/// Information in `Dict`s can be accessed with [`Dict::get`], [`Dict::range`], etc. Thanks to the
/// [`TupleCompare`] interface, any "prefix" of the record in the `Dict` can be used as indices.
/// ```
/// # use itertools::{assert_equal, Itertools};
/// # use rowantlr::utils::Dict;
/// # use rowantlr::utils::tuple::TupleBorrow;
/// let goto_table = Dict::from([
///     (0, 'a', 1),
///     (0, 'b', 2),
///     (1, 'a', 0),
///     (2, 'b', 1),
/// ]);
/// // check for emptiness
/// assert!(!goto_table.is_empty());
/// // check for key existence
/// assert!(goto_table.contains_key((&0, )));
/// assert!(!goto_table.contains_key((&1, &'b')));
/// // access the value of some specific key
/// assert_eq!(goto_table.get((&1, )), Some((&'a', &0)));
/// assert_eq!(goto_table.get((&0, &'a')), Some(&1));
/// assert_eq!(goto_table.get((&1, &'b')), None);
/// assert_eq!(goto_table.get_key_value((&1, )), Some(&(1, 'a', 0)));
/// // obtain an iterator for values of a key range or some specific key
/// assert_equal(goto_table.range((&1, )..(&2, )), [(&'a', &0)]);
/// assert_equal(goto_table.range((&1, )..=(&2, )), [(&'a', &0), (&'b', &1)]);
/// assert_equal(goto_table.equal_range((&0, )), [(&'a', &1), (&'b', &2)]);
/// // obtain an iterator for keys/values (key and value split at some index of the tuple)
/// assert_equal(goto_table.clone().into_keys::<2>(), [(0, 'a'), (0, 'b'), (1, 'a'), (2, 'b')]);
/// assert_equal(goto_table.clone().into_values::<2>(), [1, 2, 0, 1]);
/// // obtain an iterator for borrowed keys/values
/// assert_equal(goto_table.keys::<2>(), [(&0, &'a'), (&0, &'b'), (&1, &'a'), (&2, &'b')]);
/// assert_equal(goto_table.values::<2>(), [&1, &2, &0, &1]);
/// // inverse the dict (swap roles between keys and values)
/// let goto_inv = goto_table.inverse::<1>();
/// assert_eq!(goto_inv, Dict::from([
///     ('a', 0, 1),
///     ('a', 1, 0),
///     ('b', 1, 2),
///     ('b', 2, 0),
/// ]));
/// // group the values by their respective keys (key and value split at some index of the tuple)
/// let mut groups = goto_inv.groups::<1>();
/// let groups_expected = [('a', [(0, 1), (1, 0)]), ('b', [(1, 2), (2, 0)])];
/// for ((k0, g0), (k, g)) in groups.zip_eq(&groups_expected) {
///     assert_eq!(k0, k);
///     assert_equal(g0, g.iter().map(TupleBorrow::tuple_borrow));
/// }
/// ```
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq)]
#[derive(Derivative)]
#[derivative(Default(bound = ""))]
pub struct Dict<K>(Box<[K]>);

struct SingularRange<A>(A);

impl<A> RangeBounds<A> for SingularRange<A> {
    fn start_bound(&self) -> Bound<&A> { Bound::Included(&self.0) }
    fn end_bound(&self) -> Bound<&A> { Bound::Included(&self.0) }
    fn contains<U>(&self, item: &U) -> bool
        where A: PartialOrd<U>, U: ?Sized + PartialOrd<A> {
        self.0 == *item
    }
}

/// Types related to [`Dict`].
pub mod dict {
    use crate::utils::tuple::{TupleBorrow, TupleRest, TupleSplit};

    type MapTo<I, T> = std::iter::Map<I, fn(<I as Iterator>::Item) -> T>;

    type Borrowed<'a, K> = <K as TupleBorrow<'a>>::Borrowed;
    type Init<'a, K, const N: usize> = <Borrowed<'a, K> as TupleSplit<N>>::Init;
    type Tail<'a, K, const N: usize> = <Borrowed<'a, K> as TupleSplit<N>>::Tail;

    /// Double-ended iterator into a [`Dict`](super::Dict).
    pub type Iter<'a, K, Q> = MapTo<
        std::slice::Iter<'a, K>,
        <K as TupleRest<'a, Q>>::Rest
    >;

    /// Double-ended iterator for borrowed keys in a [`Dict`](super::Dict).
    pub type Keys<'a, K, const N: usize> = MapTo<std::slice::Iter<'a, K>, Init<'a, K, N>>;

    /// Double-ended iterator for borrowed values in a [`Dict`](super::Dict).
    pub type Values<'a, K, const N: usize> = MapTo<std::slice::Iter<'a, K>, Tail<'a, K, N>>;

    /// Double-ended iterator for borrowed values in a [`Dict`](super::Dict), grouped by their keys.
    pub struct Groups<'a, K, const N: usize>(pub(super) &'a [K]);

    impl<'a, K, const N: usize> Iterator for Groups<'a, K, N>
        where K: TupleBorrow<'a>, K::Borrowed: TupleSplit<N>, Init<'a, K, N>: Eq {
        type Item = (Init<'a, K, N>, Values<'a, K, N>);
        fn next(&mut self) -> Option<Self::Item> {
            if self.0.is_empty() { return None; }
            let (k0, _) = self.0[0].tuple_borrow().tuple_split();
            let n = self.0.iter()
                .map(|x| x.tuple_borrow().tuple_split().0)
                .take_while(|k| *k == k0)
                .count();
            let (init, tail) = self.0.split_at(n);
            self.0 = tail;
            Some((k0, init.iter().map(|x| x.tuple_borrow().tuple_split().1)))
        }
    }
}

impl<K> Dict<K> {
    /// Get the size of this dict.
    pub fn len(&self) -> usize { self.0.len() }

    fn locate<Q>(&self, key: Q) -> Result<usize, usize>
        where K: TupleCompare<Q> {
        self.0.binary_search_by(|a| a.tuple_compare(&key))
    }

    fn locate_lower_bound<Q>(slice: &[K], bound: Bound<&Q>) -> usize
        where Q: ?Sized, K: TupleCompare<Q> {
        match bound {
            Bound::Included(x) => slice.partition_point(|k| k.tuple_compare(x).is_lt()),
            Bound::Excluded(x) => slice.partition_point(|k| k.tuple_compare(x).is_le()),
            Bound::Unbounded => 0,
        }
    }

    fn locate_upper_bound<Q>(slice: &[K], bound: Bound<&Q>) -> usize
        where Q: ?Sized, K: TupleCompare<Q> {
        match bound {
            Bound::Included(x) => slice.partition_point(|k| k.tuple_compare(x).is_le()),
            Bound::Excluded(x) => slice.partition_point(|k| k.tuple_compare(x).is_lt()),
            Bound::Unbounded => slice.len(),
        }
    }

    /// Returns a reference to the slice in this dict.
    pub fn as_slice(&self) -> &[K] { &self.0 }

    /// Returns `true` if the dict contains a value for the specified key.
    pub fn contains_key<Q>(&self, key: Q) -> bool
        where K: TupleCompare<Q> {
        self.locate(key).is_ok()
    }

    /// Returns a reference to the value corresponding to the key.
    /// If multiple values exist, an unspecified one is returned.
    pub fn get<'a, Q>(&'a self, key: Q) -> Option<K::Rest>
        where K: TupleCompare<Q>, K: TupleRest<'a, Q> {
        let p = self.locate(key).ok()?;
        Some(self.0[p].borrow_rest())
    }

    /// Treat this dict as a classifier, by considering the keys as split points, and values as
    /// class tags for the group between the current split point (inclusive) to the next split
    /// point (exclusive). Get the class tag for some specific element.
    ///
    /// Panics if the input "underflows" the classifier, i.e. the smallest split point is still
    /// greater than the provided input, and the class tag for that specific input is therefore
    /// not well-defined.
    ///
    /// Below is an example usage of the classifiers obtained from [`Expr::freeze_char_class`]:
    /// ```
    /// # use rowantlr::utils::Dict;
    /// // auxiliary function for finding the next char:
    /// fn next_char(c: char) -> char { char::from_u32(c as u32 + 1).unwrap() }
    /// // the classifier:
    /// let chars = Dict::from([
    ///     ('\0', 0_u32),
    ///     ('$',  1), (next_char('$'),  0),
    ///     ('\'', 2), (next_char('\''), 0),
    ///     ('0',  2), (next_char('9'),  0),
    ///     ('A',  3), (next_char('Z'),  0),
    ///     ('_',  4), (next_char('_'),  0),
    ///     ('a',  4), (next_char('z'),  0),
    /// ]);
    /// // we can get the class tag for any character:
    /// assert_eq!(*chars.classify((&'x', )), 4);
    /// assert_eq!(*chars.classify((&'7', )), 2);
    /// assert_eq!(*chars.classify((&'#', )), 0);
    /// assert_eq!(*chars.classify((&'$', )), 1);
    /// ```
    ///
    /// [`Expr::freeze_char_class`]: crate::ir::lexical::Expr::freeze_char_class
    pub fn classify<'a, Q>(&'a self, key: Q) -> K::Rest
        where K: TupleCompare<Q>, K: TupleRest<'a, Q> {
        let p = Self::locate_upper_bound(&self.0, Bound::Included(&key));
        self.0[p.checked_sub(1).expect("classifier underflow")].borrow_rest()
    }

    /// Returns the key-value pair corresponding to the supplied key.
    /// If multiple entries exist, an unspecified one is returned.
    pub fn get_key_value<Q>(&self, key: Q) -> Option<&K>
        where K: TupleCompare<Q> {
        let p = self.locate(key).ok()?;
        Some(&self.0[p])
    }

    /// Creates a consuming iterator visiting all the entries, in sorted order. The dict cannot be
    /// used after calling this. The iterator element type is `K`.
    pub fn into_raw(self) -> Box<[K]> { self.0 }

    /// Creates a consuming iterator visiting all the keys, in sorted order. The dict cannot be
    /// used after calling this. The iterator element type is `K`.
    pub fn into_keys<const N: usize>(self) -> impl Iterator<Item=K::Init>
        where K: TupleSplit<N> {
        self.0.into_vec().into_iter().map(K::tuple_split).map(|(k, _)| k)
    }

    /// Creates a consuming iterator visiting all the values, in order by key. The dict cannot be
    /// used after calling this. The iterator element type is `V`.
    pub fn into_values<const N: usize>(self) -> impl Iterator<Item=K::Tail>
        where K: TupleSplit<N> {
        self.0.into_vec().into_iter().map(K::tuple_split).map(|(_, v)| v)
    }

    /// Creates an inverse of this dict: keys become values, and values become keys.
    pub fn inverse<const N: usize>(self) -> Dict<K::Rotated>
        where K: TupleRotate<N>, K::Rotated: Ord {
        self.0.into_vec().into_iter().map(K::tuple_rotate).collect()
    }

    /// Returns `true` if the dict contains no elements.
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
    /// Gets an iterator over the entries of the dict, sorted by key.
    pub fn iter(&self) -> std::slice::Iter<K> { self.0.iter() }
    /// Gets an iterator over the keys of the dict, in sorted order.
    pub fn keys<'a, const N: usize>(&'a self) -> dict::Keys<'a, K, N>
        where K: TupleBorrow<'a>, K::Borrowed: TupleSplit<N> {
        self.0.iter().map(|x| x.tuple_borrow().tuple_split().0)
    }
    /// Gets an iterator over the values of the dict, in order by key.
    pub fn values<'a, const N: usize>(&'a self) -> dict::Values<'a, K, N>
        where K: TupleBorrow<'a>, K::Borrowed: TupleSplit<N> {
        self.0.iter().map(|x| x.tuple_borrow().tuple_split().1)
    }
    /// Gets an iterator over the values of the dict, grouped by key.
    pub fn groups<const N: usize>(&self) -> dict::Groups<K, N> { dict::Groups(&self.0) }

    fn indices_range<Q, R>(&self, range: R) -> (usize, usize)
        where Q: ?Sized, K: TupleCompare<Q>, R: RangeBounds<Q> {
        let l = Self::locate_lower_bound(&self.0, range.start_bound());
        let r = Self::locate_upper_bound(&self.0, range.end_bound());
        (l, r)
    }

    /// Constructs a double-ended iterator over a sub-range of elements in the dict. The simplest
    /// way is to use the range syntax `min..max`, thus `range(min..max)` will yield elements from
    /// `min` (inclusive) to `max` (exclusive). The range may also be entered as
    /// `(Bound<T>, Bound<T>)`, so for example `range((Excluded(4), Included(10)))` will yield a
    /// left-exclusive, right-inclusive range from `4` to `10`.
    pub fn range<'a, Q, R>(&'a self, range: R) -> dict::Iter<'a, K, Q>
        where Q: ?Sized, R: RangeBounds<Q>, K: TupleCompare<Q>, K: TupleRest<'a, Q> {
        let (l, r) = self.indices_range(range);
        self.0[l..r].iter().map(K::borrow_rest)
    }

    /// Constructs a double-ended iterator for some specific key in the dict.
    pub fn equal_range<'a, Q>(&'a self, q: Q) -> dict::Iter<'a, K, Q>
        where K: TupleCompare<Q>, K: TupleRest<'a, Q> {
        self.range(SingularRange(q))
    }
}

impl<K: Ord, const N: usize> From<[K; N]> for Dict<K> {
    fn from(buffer: [K; N]) -> Self {
        Dict::from(Box::new(buffer) as Box<[K]>)
    }
}

impl<K: Ord> From<Vec<K>> for Dict<K> {
    fn from(mut buffer: Vec<K>) -> Self {
        buffer.sort_unstable();
        buffer.dedup();
        Dict(buffer.into_boxed_slice())
    }
}

impl<K: Ord> From<Box<[K]>> for Dict<K> {
    fn from(buffer: Box<[K]>) -> Self {
        Dict::from(buffer.into_vec())
    }
}

impl<K: Ord> FromIterator<K> for Dict<K> {
    fn from_iter<T: IntoIterator<Item=K>>(iter: T) -> Self {
        let vec = Vec::from_iter(iter);
        Dict::from(vec)
    }
}

impl<K> IntoIterator for Dict<K> {
    type Item = K;
    type IntoIter = std::vec::IntoIter<K>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_vec().into_iter()
    }
}

impl<'a, K> IntoIterator for &'a Dict<K> {
    type Item = &'a K;
    type IntoIter = std::slice::Iter<'a, K>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}
