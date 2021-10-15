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

use std::borrow::Borrow;
use std::iter::FromIterator;
use std::fmt::{Display, Formatter};
use std::ops::{Bound, Index, RangeBounds};

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

/// An immutable, flat dictionary.
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct Dict<K, V>(Box<[(K, V)]>);

struct SingularRange<'a, A>(&'a A);

impl<'a, A> RangeBounds<A> for SingularRange<'a, A> {
    fn start_bound(&self) -> Bound<&A> { Bound::Included(self.0) }
    fn end_bound(&self) -> Bound<&A> { Bound::Included(self.0) }
    fn contains<U>(&self, item: &U) -> bool
        where A: PartialOrd<U>, U: ?Sized + PartialOrd<A> {
        self.0 == item
    }
}

impl<K, V> Dict<K, V> {
    fn locate<Q>(&self, key: &Q) -> Result<usize, usize>
        where K: Borrow<Q>, Q: Ord + ?Sized {
        self.0.binary_search_by_key(&key, |a| a.0.borrow())
    }

    fn partition_point<F>(&self, mut pred: F) -> usize
        where F: FnMut(&K) -> bool {
        self.0.partition_point(|(k, _)| pred(k))
    }

    fn locate_lower_bound<Q, F>(&self, bound: Bound<&Q>, mut key: F) -> usize
        where Q: Ord + ?Sized, F: FnMut(&K) -> &Q {
        match bound {
            Bound::Included(x) => self.partition_point(|k| key(k) < x),
            Bound::Excluded(x) => self.partition_point(|k| key(k) <= x),
            Bound::Unbounded => 0,
        }
    }

    fn locate_upper_bound<Q, F>(&self, bound: Bound<&Q>, mut key: F) -> usize
        where Q: Ord + ?Sized, F: FnMut(&K) -> &Q {
        match bound {
            Bound::Included(x) => self.partition_point(|k| key(k) <= x),
            Bound::Excluded(x) => self.partition_point(|k| key(k) < x),
            Bound::Unbounded => self.0.len(),
        }
    }

    /// Returns a reference to the slice in this dict.
    pub fn as_slice(&self) -> &[(K, V)] { &self.0 }

    /// Returns `true` if the dict contains a value for the specified key.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
        where K: Borrow<Q>, Q: Ord + ?Sized {
        self.locate(key).is_ok()
    }

    /// Returns a reference to the value corresponding to the key.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
        where K: Borrow<Q>, Q: Ord + ?Sized {
        let p = self.locate(key).ok()?;
        Some(&self.0[p].1)
    }

    /// Returns the key-value pair corresponding to the supplied key.
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<&(K, V)>
        where K: Borrow<Q>, Q: Ord + ?Sized {
        let p = self.locate(key).ok()?;
        Some(&self.0[p])
    }

    /// Get the underlying boxed slice.
    pub fn into_raw(self) -> Box<[(K, V)]> { self.0 }

    /// Creates a consuming iterator visiting all the keys, in sorted order. The map cannot be
    /// used after calling this. The iterator element type is `K`.
    pub fn into_keys(self) -> impl Iterator<Item=K> {
        self.0.into_vec().into_iter().map(|(k, _)| k)
    }

    /// Creates a consuming iterator visiting all the values, in order by key. The map cannot be
    /// used after calling this. The iterator element type is `V`.
    pub fn into_values(self) -> impl Iterator<Item=V> {
        self.0.into_vec().into_iter().map(|(_, v)| v)
    }

    /// Returns `true` if the dict contains no elements.
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
    /// Gets an iterator over the entries of the map, sorted by key.
    pub fn iter(&self) -> std::slice::Iter<(K, V)> { self.0.iter() }
    /// Gets an iterator over the keys of the map, in sorted order.
    pub fn keys(&self) -> impl Iterator<Item=&K> { self.0.iter().map(|(k, _)| k) }
    /// Gets an iterator over the values of the map, in order by key.
    pub fn values(&self) -> impl Iterator<Item=&V> { self.0.iter().map(|(_, v)| v) }

    fn indices_range_by<T, R, F>(&self, range: R, mut key: F) -> (usize, usize)
        where T: Ord + ?Sized, R: RangeBounds<T>, F: FnMut(&K) -> &T {
        let l = self.locate_lower_bound(range.start_bound(), &mut key);
        let r = self.locate_upper_bound(range.end_bound(), &mut key);
        (l, r)
    }

    /// Constructs a double-ended iterator over a sub-range of elements in the dict. The simplest
    /// way is to use the range syntax `min..max`, thus `range(min..max)` will yield elements from
    /// `min` (inclusive) to `max` (exclusive). The range may also be entered as
    /// `(Bound<T>, Bound<T>)`, so for example `range((Excluded(4), Included(10)))` will yield a
    /// left-exclusive, right-inclusive range from `4` to `10`.
    pub fn range<T, R>(&self, range: R) -> std::slice::Iter<(K, V)>
        where T: Ord + ?Sized, R: RangeBounds<T>, K: Borrow<T> {
        self.range_by(range, K::borrow)
    }

    /// Constructs a double-ended iterator over a sub-range of elements in the dict.
    /// Instead of relying on [`Borrow`], a `key` function should be explicitly specified.
    pub fn range_by<T, R, F>(&self, range: R, key: F) -> std::slice::Iter<(K, V)>
        where T: Ord + ?Sized, R: RangeBounds<T>, F: FnMut(&K) -> &T {
        let (l, r) = self.indices_range_by(range, key);
        self.0[l..r].iter()
    }

    /// Constructs a double-ended iterator for some specific key in the dict.
    pub fn equal_range<Q, F>(&self, q: &Q) -> std::slice::Iter<(K, V)>
        where Q: Ord, K: Borrow<Q> {
        self.equal_range_by(q, K::borrow)
    }

    /// Constructs a double-ended iterator for some specific key in the dict.
    /// Instead of relying on [`Borrow`], a `key` function should be explicitly specified.
    pub fn equal_range_by<Q, F>(&self, q: &Q, key: F) -> std::slice::Iter<(K, V)>
        where Q: Ord, F: FnMut(&K) -> &Q {
        let (l, r) = self.indices_range_by(SingularRange(q), key);
        self.0[l..r].iter()
    }
}

impl<K: Ord, V> From<Vec<(K, V)>> for Dict<K, V> {
    fn from(buffer: Vec<(K, V)>) -> Self {
        Dict::from(buffer.into_boxed_slice())
    }
}

impl<K: Ord, V> From<Box<[(K, V)]>> for Dict<K, V> {
    fn from(mut buffer: Box<[(K, V)]>) -> Self {
        buffer.sort_unstable_by(|(k1, _), (k2, _)| k1.cmp(k2));
        Dict(buffer)
    }
}

impl<K: Ord, V> FromIterator<(K, V)> for Dict<K, V> {
    fn from_iter<T: IntoIterator<Item=(K, V)>>(iter: T) -> Self {
        let vec = Vec::from_iter(iter);
        Dict::from(vec)
    }
}

impl<K, V> IntoIterator for Dict<K, V> {
    type Item = (K, V);
    type IntoIter = std::vec::IntoIter<(K, V)>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_vec().into_iter()
    }
}

impl<'a, K, V> IntoIterator for &'a Dict<K, V> {
    type Item = &'a (K, V);
    type IntoIter = std::slice::Iter<'a, (K, V)>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl<K, Q, V> Index<&'_ Q> for Dict<K, V>
    where K: Borrow<Q> + Ord, Q: Ord + ?Sized {
    type Output = V;
    #[inline(always)]
    fn index(&self, index: &'_ Q) -> &V {
        self.get(index).unwrap()
    }
}
