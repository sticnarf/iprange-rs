//! `iprange` is a library for managing IP ranges.
//!
//! An [`IpRange`] is a set of networks.
//! You can add or remove a [`Ipv4Net`] from an [`IpRange`].
//!
//! It also supports these useful operations:
//!
//! * [`merge`]
//! * [`intersect`]
//! * [`exclude`]
//!
//! Here is a simple example:
//!
//! ```
//! extern crate iprange;
//! extern crate ipnet;
//!
//! use std::net::Ipv4Addr;
//! use iprange::IpRange;
//! use ipnet::Ipv4Net;
//!
//! fn main() {
//!     let ip_range: IpRange<Ipv4Net> = ["10.0.0.0/8", "172.16.0.0/16", "192.168.1.0/24"]
//!         .iter()
//!         .map(|s| s.parse().unwrap())
//!         .collect();
//!
//!     assert!(ip_range.contains(&"172.16.32.1".parse::<Ipv4Addr>().unwrap()));
//!     assert!(ip_range.contains(&"192.168.1.1".parse::<Ipv4Addr>().unwrap()));
//! }
//! ```
//!
//! Currently, this library supports IPv4 only.
//!
//! [`IpRange`]: struct.IpRange.html
//! [`Ipv4Net`]: https://docs.rs/ipnet/1.0.0/ipnet/struct.Ipv4Net.html
//! [`merge`]: struct.IpRange.html#method.merge
//! [`intersect`]: struct.IpRange.html#method.intersect
//! [`exclude`]: struct.IpRange.html#method.exclude

extern crate ipnet;

use std::net::Ipv4Addr;
use std::rc::Rc;
use std::cell::RefCell;
use std::iter::FromIterator;
use std::collections::VecDeque;
use std::marker::{Sized, PhantomData};
use ipnet::Ipv4Net;

/// A set of networks that supports various operations:
///
/// * [`add`]
/// * [`remove`]
/// * [`contains`]
/// * [`merge`]
/// * [`intersect`]
/// * [`exclude`]
///
/// `IntoIter` is implemented for `&IpRange`. So, you can use `for`
/// to iterate over the networks in an `IpRange`:
///
/// ```
/// extern crate ipnet;
/// extern crate iprange;
///
/// use iprange::IpRange;
/// use ipnet::Ipv4Net;
///
/// fn main() {
///     let ip_range: IpRange<Ipv4Net> = ["172.16.0.0/16", "192.168.1.0/24"]
///         .iter()
///         .map(|s| s.parse().unwrap())
///         .collect();
///
///     for network in &ip_range {
///         println!("{:?}", network);
///     }
/// }
/// ```
///
/// [`add`]: struct.IpRange.html#method.add
/// [`remove`]: struct.IpRange.html#method.remove
/// [`contains`]: struct.IpRange.html#method.contains
/// [`merge`]: struct.IpRange.html#method.merge
/// [`intersect`]: struct.IpRange.html#method.intersect
/// [`exclude`]: struct.IpRange.html#method.exclude
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IpRange<R>
    where R: IpNet + ToNetwork<R> + Clone
{
    // IpRange uses a radix trie to store networks
    trie: IpTrie<R>,
    phantom_net: PhantomData<R>,
}

impl<R> IpRange<R>
    where R: IpNet + ToNetwork<R> + Clone
{
    /// Creates an empty `IpRange`.
    pub fn new() -> IpRange<R> {
        IpRange {
            trie: IpTrie::new(),
            phantom_net: PhantomData,
        }
    }

    /// Add a network to `self`.
    ///
    /// Returns `&mut self` in order to enable method chaining.
    ///
    /// Pay attention that this operation will not combine two
    /// networks automatically. To do this, call [`simplify`] method
    /// explicitly. For example:
    ///
    /// ```
    /// extern crate iprange;
    /// extern crate ipnet;
    ///
    /// use iprange::IpRange;
    /// use ipnet::Ipv4Net;
    ///
    /// fn main() {
    ///     let mut ip_range: IpRange<Ipv4Net> = IpRange::new();
    ///     ip_range.add("192.168.0.0/24".parse().unwrap())
    ///            .add("192.168.1.0/24".parse().unwrap());
    ///     assert_eq!(ip_range.into_iter().count(), 2);
    ///
    ///     ip_range.simplify();
    ///     assert_eq!(ip_range.into_iter().count(), 1);
    /// }
    /// ```
    ///
    /// [`simplify`]: struct.IpRange.html#method.simplify
    pub fn add(&mut self, network: R) -> &mut IpRange<R> {
        self.trie.insert(network);
        self
    }

    /// Remove a network from `self`.
    ///
    /// Returns `&mut self` in order to enable method chaining.
    ///
    /// `self` does not necessarily has exactly the network to be removed.
    /// The network can be a networkwork of a network in `self`.
    /// This method will do splitting and remove the corresponding network.
    /// For example:
    ///
    /// ```
    /// extern crate iprange;
    /// extern crate ipnet;
    ///
    /// use iprange::IpRange;
    /// use ipnet::Ipv4Net;
    ///
    /// fn main() {
    ///     let mut ip_range: IpRange<Ipv4Net> = IpRange::new();
    ///     ip_range.add("192.168.0.0/23".parse().unwrap())
    ///             .remove("192.168.0.0/24".parse().unwrap());
    ///     // Now, ip_range has only one network: "192.168.1.0/24".
    /// }
    /// ```
    pub fn remove(&mut self, network: R) -> &mut IpRange<R> {
        self.trie.remove(network);
        self
    }

    /// Simplify `self` by combining networks. For example:
    ///
    /// ```
    /// extern crate iprange;
    /// extern crate ipnet;
    ///
    /// use iprange::IpRange;
    /// use ipnet::Ipv4Net;
    ///
    /// fn main() {
    ///     let mut ip_range: IpRange<Ipv4Net> = IpRange::new();
    ///     ip_range
    ///         .add("192.168.0.0/20".parse().unwrap())
    ///         .add("192.168.16.0/22".parse().unwrap())
    ///         .add("192.168.20.0/24".parse().unwrap())
    ///         .add("192.168.21.0/24".parse().unwrap())
    ///         .add("192.168.22.0/24".parse().unwrap())
    ///         .add("192.168.23.0/24".parse().unwrap())
    ///         .add("192.168.24.0/21".parse().unwrap())
    ///         .simplify();
    ///     // Now, ip_range has only one network: "192.168.0.0/19".
    /// }
    /// ```
    pub fn simplify(&mut self) {
        self.trie.simplify();
    }

    /// Returns a new `IpRange` which contains all networks
    /// that is either in `self` or in `other`.
    ///
    /// The returned `IpRange` is simplified.
    pub fn merge(&self, other: &IpRange<R>) -> IpRange<R> {
        self.into_iter().chain(other.into_iter()).collect()
    }

    /// Returns a new `IpRange` which contains all networks
    /// that is in both `self` and `other`.
    ///
    /// The returned `IpRange` is simplified.
    pub fn intersect(&self, other: &IpRange<R>) -> IpRange<R> {
        let range1 = self.into_iter().filter(|network| other.contains(network));
        let range2 = other.into_iter().filter(|network| self.contains(network));
        range1.chain(range2).collect()
    }

    /// Returns a new `IpRange` which contains all networks
    /// that is in `self` while not in `other`.
    ///
    /// The returned `IpRange` is simplified.
    pub fn exclude(&self, other: &IpRange<R>) -> IpRange<R> {
        let mut new = (*self).clone();
        for network in other {
            new.remove(network);
        }
        new
    }

    /// Tests if `self` contains `network`.
    ///
    /// `network` is anything that can be converted into `R`.
    /// See `ToNetwork<R>` for detail.
    pub fn contains<T: ToNetwork<R>>(&self, network: &T) -> bool {
        self.supernet(&network.to_network()).is_some()
    }

    /// Returns the network in `self` which is the supernetwork of `network`.
    ///
    /// Returns None if no network in `self` contains `network`.
    pub fn supernet<T: ToNetwork<R>>(&self, network: &T) -> Option<R> {
        self.trie.search(network.to_network())
    }
}

const HIGHEST_ONE_U32: u32 = 1 << 31;

impl<'a, R> IntoIterator for &'a IpRange<R>
    where R: IpNet + ToNetwork<R> + Clone
{
    type Item = R;
    type IntoIter = IpRangeIter<R>;

    fn into_iter(self) -> Self::IntoIter {
        let mut queue = VecDeque::new();
        if let Some(root) = self.trie.root.as_ref() {
            queue.push_back(R::S::init(root.clone()));
        }
        IpRangeIter { queue }
    }
}

/// Anything that can be converted to `IpNet`.
///
/// Due to limitation of Rust's type system,
/// this trait is only implemented for some
/// concrete types.
pub trait ToNetwork<R: IpNet> {
    fn to_network(&self) -> R;
}

impl ToNetwork<Ipv4Net> for Ipv4Net {
    #[inline]
    fn to_network(&self) -> Ipv4Net {
        self.trunc()
    }
}

impl ToNetwork<Ipv4Net> for Ipv4Addr {
    #[inline]
    fn to_network(&self) -> Ipv4Net {
        Ipv4Net::new(*self, 32).unwrap()
    }
}

impl ToNetwork<Ipv4Net> for u32 {
    #[inline]
    fn to_network(&self) -> Ipv4Net {
        Ipv4Net::new((*self).into(), 32).unwrap()
    }
}

impl ToNetwork<Ipv4Net> for [u8; 4] {
    #[inline]
    fn to_network(&self) -> Ipv4Net {
        Ipv4Net::new((*self).into(), 32).unwrap()
    }
}

/// An iterator over the networks in an [`IpRange`].
///
/// BFS (Breadth-First-Search) is used for traversing the inner Radix Trie.
///
/// [`IpRange`]: struct.IpRange.html
pub struct IpRangeIter<R>
    where R: IpNet
{
    queue: VecDeque<R::S>,
}

pub trait TraverseState<R>
    where R: IpNet
{
    fn node(&self) -> Rc<RefCell<IpTrieNode>>;

    fn init(root: Rc<RefCell<IpTrieNode>>) -> Self;

    fn transit(&self, next_node: Rc<RefCell<IpTrieNode>>, current_bit: bool) -> Self;

    fn build(&self) -> R;
}

pub struct Ipv4TraverseState
{
    node: Rc<RefCell<IpTrieNode>>,
    prefix: u32,
    prefix_len: u32
}

impl TraverseState<Ipv4Net> for Ipv4TraverseState {
    fn node(&self) -> Rc<RefCell<IpTrieNode>> {
        self.node.clone()
    }

    fn init(root: Rc<RefCell<IpTrieNode>>) -> Self {
        Ipv4TraverseState {
            node: root,
            prefix: 0,
            prefix_len: 0
        }
    }

    fn transit(&self, next_node: Rc<RefCell<IpTrieNode>>, current_bit: bool) -> Self {
        let mask = if current_bit { HIGHEST_ONE_U32 >> self.prefix_len } else { 0 };
        Ipv4TraverseState {
            node: next_node,
            prefix: self.prefix | mask,
            prefix_len: self.prefix_len + 1
        }
    }

    fn build(&self) -> Ipv4Net {
        Ipv4Net::new(self.prefix.into(), self.prefix_len as u8).unwrap()
    }
}

impl<R> Iterator for IpRangeIter<R>
    where R: IpNet
{
    type Item = R;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(elem) = self.queue.pop_front() {
            // Get the front element of the queue.
            // If it is a leaf, it represents a network
            let node = elem.node();
            if node.borrow().is_leaf() {
                return Some(elem.build());
            }
            [0, 1].iter().for_each(|&i| {
                if let Some(child) = node.borrow().children[i as usize].as_ref() {
                    // Push the child nodes into the queue
                    self.queue.push_back(elem.transit(child.clone(), i != 0));
                }
            });
        }
        None
    }
}

impl<R> FromIterator<R> for IpRange<R>
    where R: IpNet + ToNetwork<R> + Clone
{
    fn from_iter<T>(iter: T) -> Self
        where T: IntoIterator<Item=R>,
    {
        let mut ip_range = IpRange::new();
        for network in iter {
            ip_range.add(network);
        }
        ip_range.simplify();
        ip_range
    }
}

pub trait IpNet
    where Self: Sized
{
    type S: TraverseState<Self>;
    type I: Iterator<Item=bool>;

    fn prefix_bits(&self) -> Self::I;

    fn prefix_len(&self) -> u8;

    fn with_new_prefix(&self, len: u8) -> Self;
}

pub struct Ipv4PrefixBitIterator {
    prefix: u32,
    prefix_len: u8
}

impl Iterator for Ipv4PrefixBitIterator {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.prefix_len > 0 {
            let prefix = self.prefix;
            self.prefix <<= 1;
            self.prefix_len -= 1;
            Some(prefix & HIGHEST_ONE_U32 != 0)
        } else {
            None
        }
    }
}

impl IpNet for Ipv4Net {
    type S = Ipv4TraverseState;
    type I = Ipv4PrefixBitIterator;

    #[inline]
    fn prefix_bits(&self) -> Self::I {
        let prefix: u32 = self.addr().into();
        Ipv4PrefixBitIterator {
            prefix,
            prefix_len: self.prefix_len()
        }
    }

    #[inline]
    fn prefix_len(&self) -> u8 {
        self.prefix_len()
    }

    #[inline]
    fn with_new_prefix(&self, len: u8) -> Self {
        Ipv4Net::new(self.addr(), len).unwrap().trunc()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct IpTrie<R>
    where R: IpNet
{
    root: Option<Rc<RefCell<IpTrieNode>>>,
    phantom_net: PhantomData<R>
}

impl<R> IpTrie<R>
    where R: IpNet
{
    fn new() -> IpTrie<R> {
        IpTrie {
            root: None,
            phantom_net: PhantomData,
        }
    }

    fn insert(&mut self, network: R) {
        if self.root.is_none() {
            self.root = Some(Rc::new(RefCell::new(IpTrieNode::new())))
        }

        let mut node = self.root.clone().unwrap(); // The current node

        let bits = network.prefix_bits();
        for bit in bits {
            let i = bit as usize;
            let child = node.borrow().children[i].clone();
            match child {
                Some(child) => {
                    if child.borrow().is_leaf() {
                        // This means the network to be inserted
                        // is already in the trie.
                        return;
                    }
                    node = child;
                }
                None => {
                    let new_node = Rc::new(RefCell::new(IpTrieNode::new()));
                    (*node.borrow_mut()).children[i] = Some(new_node.clone());
                    node = new_node;
                }
            }
        }
        (*node.borrow_mut()).children = [None, None];
    }

    fn search(&self, network: R) -> Option<R> {
        if self.root.is_none() {
            return None;
        }
        let mut node = self.root.clone().unwrap();

        let bits = network.prefix_bits();
        for (j, bit) in bits.into_iter().enumerate() {
            if node.borrow().is_leaf() {
                return Some(network.with_new_prefix(j as u8));
            }

            let i = bit as usize;
            let child = node.borrow().children[i].clone();
            match child {
                Some(child) => node = child,
                None => return None,
            }
        }

        if node.borrow().is_leaf() {
            Some(network)
        } else {
            None
        }

        // The commented code below is more clear. However, this uses a
        // commented method `search` in IpTrieNode, and the performance
        // is relatively poorer that the implementation above.

        // self.root.as_ref().and_then(|root| {
        //     let mut bits = network.prefix_bits();
        //     let first_bit = bits.next();
        //     root.borrow()
        //         .search(bits, first_bit, 0)
        //         .map(|prefix_size| {
        //             network.with_new_prefix(prefix_size)
        //         })
        // })
    }

    fn remove(&mut self, network: R) {
        let root = self.root.clone();
        if let Some(root) = root.as_ref() {
            let mut bits = network.prefix_bits();
            match bits.next() {
                Some(next_bit) => root.borrow_mut().remove(bits, next_bit),
                None => self.root = None // Reinitialize the trie
            }
        }
    }

    fn simplify(&mut self) {
        if let Some(root) = self.root.as_ref() {
            root.borrow_mut().simplify();
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IpTrieNode {
    children: [Option<Rc<RefCell<IpTrieNode>>>; 2],
}

impl IpTrieNode {
    fn new() -> IpTrieNode {
        IpTrieNode {
            children: [None, None],
        }
    }

    // If both the zero child and the one child of a node are None,
    // it is a leaf node, and it represents a network whose
    // prefix is the path from root to it.
    #[inline]
    fn is_leaf(&self) -> bool {
        self.children[0].is_none() && self.children[1].is_none()
    }

    // If the two children of a node are all leaf node,
    // they can be merged into a new leaf node.
    fn simplify(&mut self) {
        let leaf_count: u32 = self.children
            .iter()
            .map(|child| {
                child
                    .as_ref()
                    .map(|child| {
                        child.borrow_mut().simplify();
                        child.borrow().is_leaf() as u32
                    })
                    .unwrap_or_default()
            })
            .sum();
        if leaf_count == 2 {
            self.children = [None, None];
        }
    }

    //    fn search<I>(&self, mut bits: I, current_bit: Option<bool>, acc: u8) -> Option<u8>
    //        where I: Iterator<Item=bool>
    //    {
    //        if self.is_leaf() {
    //            Some(acc)
    //        } else {
    //            if let Some(current_bit) = current_bit {
    //                if let Some(child) = self.children[current_bit as usize].clone() {
    //                    let next_bit = bits.next();
    //                    return child
    //                        .borrow_mut()
    //                        .search(bits, next_bit, acc + 1);
    //                }
    //            }
    //            None
    //        }
    //    }

    fn remove<I>(&mut self, mut bits: I, current_bit: bool)
        where I: Iterator<Item=bool>
    {
        let i = current_bit as usize;
        let next_bit = bits.next();

        // If the current node is a leaf node, and we have a network
        // to remove, we must split it into two deeper nodes.
        if self.is_leaf() {
            self.children = [
                Some(Rc::new(RefCell::new(IpTrieNode::new()))),
                Some(Rc::new(RefCell::new(IpTrieNode::new()))),
            ];
        }

        match next_bit {
            Some(next_bit) => if let Some(child) = self.children[i].clone() {
                // Remove the deeper node recursively
                child.borrow_mut().remove(bits, next_bit);

                // In general, a leaf node represents a complete
                // network. However, the child node cannot be a complete
                // network after removing a network from it.
                // This occurring indicates the only child of the
                // child node is removed, and now this child node
                // should be marked None.
                if child.borrow().is_leaf() {
                    self.children[i] = None;
                }
            },
            None => {
                // Remove the node that represents the network.
                self.children[i] = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_invalid_networks() {
        assert!("192.168.256.130/5".parse::<Ipv4Net>().is_err());
        assert!("192.168.5.130/-1".parse::<Ipv4Net>().is_err());
        assert!("192.168.5.130/33".parse::<Ipv4Net>().is_err());
        assert!("192.168.5.33".parse::<Ipv4Net>().is_err());
        assert!("192.168.5.130/0.0.0".parse::<Ipv4Net>().is_err());
        assert!("192.168.5.130/0.0.0.256".parse::<Ipv4Net>().is_err());
    }

    impl IpRange<Ipv4Net>
    {
        fn get_network(&self, prefix_size: usize, prefix: &str) -> Option<Ipv4Net> {
            self.trie
                .search(format!("{}/{}", prefix, prefix_size).parse().unwrap())
        }
    }

    #[test]
    fn add_single_network() {
        let mut ip_range = IpRange::new();
        let network = "192.168.5.0/24".parse().unwrap();
        ip_range.add(network);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network), ip_range.get_network(24, "192.168.5.0"));
    }

    #[test]
    fn add_multiple_networks_disjoint() {
        let mut ip_range = IpRange::new();
        let network1 = "10.0.0.0/8".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        let network3 = "192.168.1.0/24".parse().unwrap();
        let network4 = "254.254.254.254/32".parse().unwrap();
        ip_range
            .add(network1)
            .add(network2)
            .add(network3)
            .add(network4)
            .simplify();

        assert_eq!(ip_range.into_iter().count(), 4);
        assert_eq!(Some(network1), ip_range.get_network(8, "10.0.0.0"));
        assert_eq!(Some(network2), ip_range.get_network(16, "172.16.0.0"));
        assert_eq!(Some(network3), ip_range.get_network(24, "192.168.1.0"));
        assert_eq!(Some(network4), ip_range.get_network(32, "254.254.254.254"));
    }


    #[test]
    fn simplify() {
        let mut ip_range = IpRange::new();
        ip_range
            .add("192.168.0.0/20".parse().unwrap())
            .add("192.168.16.0/22".parse().unwrap())
            .add("192.168.20.0/24".parse().unwrap())
            .add("192.168.21.0/24".parse().unwrap())
            .add("192.168.22.0/24".parse().unwrap())
            .add("192.168.23.0/24".parse().unwrap())
            .add("192.168.24.0/21".parse().unwrap())
            .simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(
            "192.168.0.0/19".parse().ok(),
            ip_range.get_network(19, "192.168.0.0")
        );
    }


    #[test]
    fn add_multiple_networks_joint1() {
        let mut ip_range = IpRange::new();
        let network1 = "172.16.4.0/24".parse().unwrap();
        let network2 = "172.16.4.0/22".parse().unwrap();
        ip_range.add(network1).add(network2).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(22, "172.16.4.0"));
    }

    #[test]
    fn add_multiple_networks_joint2() {
        let mut ip_range = IpRange::new();
        let network1 = "172.16.5.0/24".parse().unwrap();
        let network2 = "172.16.4.0/22".parse().unwrap();
        ip_range.add(network1).add(network2).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(22, "172.16.4.0"));
    }

    #[test]
    fn add_multiple_networks_joint3() {
        let mut ip_range = IpRange::new();
        let network1 = "172.16.4.0/24".parse().unwrap();
        let network2 = "172.16.4.0/22".parse().unwrap();
        ip_range.add(network2).add(network1).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(22, "172.16.4.0"));
    }

    #[test]
    fn add_multiple_networks_joint4() {
        let mut ip_range = IpRange::new();
        let network1 = "172.16.5.0/24".parse().unwrap();
        let network2 = "172.16.5.0/24".parse().unwrap();
        ip_range.add(network1).add(network2).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(24, "172.16.5.0"));
    }

    #[test]
    fn add_multiple_networks_joint5() {
        let mut ip_range = IpRange::new();
        let network1 = "172.16.5.0/24".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        ip_range.add(network1).add(network2).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(16, "172.16.0.0"));
    }

    #[test]
    fn add_multiple_networks_joint6() {
        let mut ip_range = IpRange::new();
        let network1 = "172.16.5.0/24".parse().unwrap();
        let network2 = "0.0.0.0/0".parse().unwrap();
        ip_range.add(network1).add(network2).simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(0, "0.0.0.0"));
    }

    #[test]
    fn remove_networks_no_split() {
        let mut ip_range = IpRange::new();
        let network1 = "192.168.0.0/24".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        ip_range.add(network1).add(network2).simplify();

        ip_range.remove(network1);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(16, "172.16.0.0"));
    }

    #[test]
    fn remove_networks_split1() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/22".parse().unwrap());
        ip_range
            .remove("192.168.2.0/23".parse().unwrap())
            .simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(
            Some("192.168.0.0/23".parse().unwrap()),
            ip_range.get_network(23, "192.168.0.0")
        );
    }

    #[test]
    fn remove_networks_split2() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/22".parse().unwrap());
        ip_range
            .remove("192.168.0.0/23".parse().unwrap())
            .simplify();

        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(
            Some("192.168.2.0/23".parse().unwrap()),
            ip_range.get_network(23, "192.168.2.0")
        );
    }

    #[test]
    fn remove_networks_split3() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/22".parse().unwrap());
        ip_range
            .remove("192.168.2.0/25".parse().unwrap())
            .simplify();

        assert_eq!(ip_range.into_iter().count(), 3);
        assert_eq!(
            Some("192.168.0.0/23".parse().unwrap()),
            ip_range.get_network(23, "192.168.0.0")
        );
        assert_eq!(
            Some("192.168.2.128/25".parse().unwrap()),
            ip_range.get_network(25, "192.168.2.128")
        );
        assert_eq!(
            Some("192.168.3.0/24".parse().unwrap()),
            ip_range.get_network(24, "192.168.3.0")
        );
    }


    impl IpRange<Ipv4Net>
    {
        fn contains_ip(&self, ip: &str) -> bool {
            self.contains(&ip.parse::<Ipv4Addr>().unwrap())
        }

        fn find_network_by_ip(&self, ip: &str) -> Option<Ipv4Net> {
            self.supernet(&ip.parse::<Ipv4Addr>().unwrap())
        }

        fn contains_network(&self, network: &str) -> bool {
            self.contains(&network.parse::<Ipv4Net>().unwrap())
        }

        fn super_network_by_network(&self, network: &str) -> Option<Ipv4Net> {
            self.supernet(&network.parse::<Ipv4Net>().unwrap())
        }
    }

    #[test]
    fn contains_ip_with_one_network() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/24".parse().unwrap());

        assert!(ip_range.contains_ip("192.168.0.0"));
        assert!(ip_range.contains_ip("192.168.0.128"));
        assert!(ip_range.contains_ip("192.168.0.255"));
        assert!(!ip_range.contains_ip("192.167.255.255"));
        assert!(!ip_range.contains_ip("192.168.1.0"));
    }

    #[test]
    fn contains_ip_with_many_networks() {
        let mut ip_range = IpRange::new();
        ip_range
            .add("192.168.0.0/24".parse().unwrap())
            .add("172.16.0.0/16".parse().unwrap())
            .add("10.0.0.0/8".parse().unwrap())
            .simplify();

        assert!(ip_range.contains_ip("192.168.0.128"));
        assert!(ip_range.contains_ip("172.16.32.1"));
        assert!(ip_range.contains_ip("10.10.10.10"));
        assert!(!ip_range.contains_ip("0.0.0.0"));
        assert!(!ip_range.contains_ip("8.8.8.8"));
        assert!(!ip_range.contains_ip("11.0.0.0"));
        assert!(!ip_range.contains_ip("192.167.255.255"));
        assert!(!ip_range.contains_ip("255.255.255.255"));
    }

    #[test]
    fn contains_ip_boundary1() {
        let mut ip_range = IpRange::new();
        ip_range.add("0.0.0.0/0".parse().unwrap());

        assert!(ip_range.contains_ip("0.0.0.0"));
        assert!(ip_range.contains_ip("8.8.8.8"));
        assert!(ip_range.contains_ip("192.168.0.0"));
        assert!(ip_range.contains_ip("192.168.1.1"));
    }

    #[test]
    fn contains_ip_boundary2() {
        let mut ip_range = IpRange::new();
        ip_range.add("254.254.254.254/32".parse().unwrap());

        assert!(!ip_range.contains_ip("0.0.0.0"));
        assert!(!ip_range.contains_ip("8.8.8.8"));
        assert!(!ip_range.contains_ip("192.168.0.0"));
        assert!(ip_range.contains_ip("254.254.254.254"));
    }

    #[test]
    fn find_network_with_one_network() {
        let mut ip_range = IpRange::new();
        let network = "192.168.0.0/24".parse().unwrap();
        ip_range.add(network);

        assert_eq!(Some(network), ip_range.find_network_by_ip("192.168.0.0"));
        assert_eq!(Some(network), ip_range.find_network_by_ip("192.168.0.128"));
        assert_eq!(Some(network), ip_range.find_network_by_ip("192.168.0.255"));
        assert_eq!(None, ip_range.find_network_by_ip("192.167.255.255"));
        assert_eq!(None, ip_range.find_network_by_ip("192.168.1.0"));
    }

    #[test]
    fn find_network_with_many_networks() {
        let mut ip_range = IpRange::new();
        let network1 = "192.168.0.0/24".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        let network3 = "10.0.0.0/8".parse().unwrap();
        ip_range
            .add(network1)
            .add(network2)
            .add(network3)
            .simplify();

        assert_eq!(Some(network1), ip_range.find_network_by_ip("192.168.0.128"));
        assert_eq!(Some(network2), ip_range.find_network_by_ip("172.16.32.1"));
        assert_eq!(Some(network3), ip_range.find_network_by_ip("10.10.10.10"));
        assert_eq!(None, ip_range.find_network_by_ip("0.0.0.0"));
        assert_eq!(None, ip_range.find_network_by_ip("8.8.8.8"));
        assert_eq!(None, ip_range.find_network_by_ip("11.0.0.0"));
        assert_eq!(None, ip_range.find_network_by_ip("192.167.255.255"));
        assert_eq!(None, ip_range.find_network_by_ip("255.255.255.255"));
    }

    #[test]
    fn find_network_boundary1() {
        let mut ip_range = IpRange::new();
        let network = "0.0.0.0/0".parse().unwrap();
        ip_range.add(network);

        assert_eq!(Some(network), ip_range.find_network_by_ip("0.0.0.0"));
        assert_eq!(Some(network), ip_range.find_network_by_ip("8.8.8.8"));
        assert_eq!(Some(network), ip_range.find_network_by_ip("192.168.0.0"));
        assert_eq!(Some(network), ip_range.find_network_by_ip("192.168.1.1"));
    }

    #[test]
    fn find_network_boundary2() {
        let mut ip_range = IpRange::new();
        let network = "254.254.254.254/32".parse().unwrap();
        ip_range.add(network);

        assert_eq!(None, ip_range.find_network_by_ip("0.0.0.0"));
        assert_eq!(None, ip_range.find_network_by_ip("8.8.8.8"));
        assert_eq!(None, ip_range.find_network_by_ip("192.168.0.0"));
        assert_eq!(
            Some(network),
            ip_range.find_network_by_ip("254.254.254.254")
        );
    }

    #[test]
    fn contains_network_with_one_network() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/24".parse().unwrap());

        assert!(ip_range.contains_network("192.168.0.0/24"));
        assert!(ip_range.contains_network("192.168.0.128/25"));
        assert!(!ip_range.contains_network("192.168.0.0/23"));
        assert!(!ip_range.contains_network("192.168.1.0/24"));
        assert!(!ip_range.contains_network("192.167.0.0/24"));
    }

    #[test]
    fn contains_network_with_many_networks() {
        let mut ip_range = IpRange::new();
        ip_range
            .add("192.168.0.0/24".parse().unwrap())
            .add("172.16.0.0/16".parse().unwrap())
            .add("10.0.0.0/8".parse().unwrap())
            .simplify();

        assert!(ip_range.contains_network("192.168.0.128/25"));
        assert!(ip_range.contains_network("172.16.32.0/20"));
        assert!(ip_range.contains_network("10.10.0.0/16"));
        assert!(!ip_range.contains_network("0.0.0.0/0"));
        assert!(!ip_range.contains_network("8.0.0.0/6"));
        assert!(!ip_range.contains_network("8.0.0.0/7"));
        assert!(!ip_range.contains_network("11.0.0.0/9"));
        assert!(!ip_range.contains_network("192.167.255.255/32"));
        assert!(!ip_range.contains_network("255.0.0.0/8"));
    }

    #[test]
    fn contains_network_boundary1() {
        let mut ip_range = IpRange::new();
        ip_range.add("0.0.0.0/0".parse().unwrap());

        assert!(ip_range.contains_network("0.0.0.0/0"));
        assert!(ip_range.contains_network("8.0.0.0/6"));
        assert!(ip_range.contains_network("11.0.0.0/9"));
        assert!(ip_range.contains_network("192.168.0.128/25"));
        assert!(ip_range.contains_network("255.255.255.255/32"));
    }

    #[test]
    fn contains_network_boundary2() {
        let mut ip_range = IpRange::new();
        ip_range.add("254.254.254.254/32".parse().unwrap());

        assert!(!ip_range.contains_network("0.0.0.0/0"));
        assert!(!ip_range.contains_network("8.0.0.0/6"));
        assert!(!ip_range.contains_network("254.254.0.0/16"));
        assert!(ip_range.contains_network("254.254.254.254/32"));
        assert!(!ip_range.contains_network("255.255.255.255/32"));
    }

    #[test]
    fn super_network_with_one_network() {
        let mut ip_range = IpRange::new();
        let network = "192.168.0.0/24".parse().unwrap();
        ip_range.add(network);

        assert_eq!(
            Some(network),
            ip_range.super_network_by_network("192.168.0.0/24")
        );
        assert_eq!(
            Some(network),
            ip_range.super_network_by_network("192.168.0.128/25")
        );
        assert_eq!(None, ip_range.super_network_by_network("192.168.0.0/23"));
        assert_eq!(None, ip_range.super_network_by_network("192.168.1.0/24"));
        assert_eq!(None, ip_range.super_network_by_network("192.167.0.0/24"));
    }

    #[test]
    fn super_network_with_many_networks() {
        let mut ip_range = IpRange::new();
        let network1 = "192.168.0.0/24".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        let network3 = "10.0.0.0/8".parse().unwrap();
        ip_range
            .add(network1)
            .add(network2)
            .add(network3)
            .simplify();

        assert_eq!(
            Some(network1),
            ip_range.super_network_by_network("192.168.0.128/25")
        );
        assert_eq!(
            Some(network2),
            ip_range.super_network_by_network("172.16.32.0/20")
        );
        assert_eq!(
            Some(network3),
            ip_range.super_network_by_network("10.10.0.0/16")
        );
        assert_eq!(None, ip_range.super_network_by_network("0.0.0.0/0"));
        assert_eq!(None, ip_range.super_network_by_network("8.0.0.0/6"));
        assert_eq!(None, ip_range.super_network_by_network("8.0.0.0/7"));
        assert_eq!(None, ip_range.super_network_by_network("11.0.0.0/9"));
        assert_eq!(
            None,
            ip_range.super_network_by_network("192.167.255.255/32")
        );
        assert_eq!(None, ip_range.super_network_by_network("255.0.0.0/8"));
    }

    #[test]
    fn super_network_boundary1() {
        let mut ip_range = IpRange::new();
        let network = "0.0.0.0/0".parse().unwrap();
        ip_range.add(network);

        assert_eq!(
            Some(network),
            ip_range.super_network_by_network("0.0.0.0/0")
        );
        assert_eq!(
            Some(network),
            ip_range.super_network_by_network("8.0.0.0/6")
        );
        assert_eq!(
            Some(network),
            ip_range.super_network_by_network("11.0.0.0/9")
        );
        assert_eq!(
            Some(network),
            ip_range.super_network_by_network("192.168.0.128/25")
        );
        assert_eq!(
            Some(network),
            ip_range.super_network_by_network("255.255.255.255/32")
        );
    }

    #[test]
    fn super_network_boundary2() {
        let mut ip_range = IpRange::new();
        let network = "254.254.254.254/32".parse().unwrap();
        ip_range.add(network);

        assert_eq!(None, ip_range.super_network_by_network("0.0.0.0/0"));
        assert_eq!(None, ip_range.super_network_by_network("8.0.0.0/6"));
        assert_eq!(None, ip_range.super_network_by_network("254.254.0.0/16"));
        assert_eq!(
            Some(network),
            ip_range.super_network_by_network("254.254.254.254/32")
        );
        assert_eq!(
            None,
            ip_range.super_network_by_network("255.255.255.255/32")
        );
    }

    #[test]
    fn merge_empty1() {
        let ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "10.0.0.0/8".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        let network3 = "192.168.1.0/24".parse().unwrap();
        let network4 = "254.254.254.254/32".parse().unwrap();
        ip_range2
            .add(network1)
            .add(network2)
            .add(network3)
            .add(network4)
            .simplify();

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 4);
        assert_eq!(Some(network1), ip_range.get_network(8, "10.0.0.0"));
        assert_eq!(Some(network2), ip_range.get_network(16, "172.16.0.0"));
        assert_eq!(Some(network3), ip_range.get_network(24, "192.168.1.0"));
        assert_eq!(Some(network4), ip_range.get_network(32, "254.254.254.254"));
    }

    #[test]
    fn merge_empty2() {
        let mut ip_range1 = IpRange::new();
        let ip_range2 = IpRange::new();
        let network1 = "10.0.0.0/8".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        let network3 = "192.168.1.0/24".parse().unwrap();
        let network4 = "254.254.254.254/32".parse().unwrap();
        ip_range1
            .add(network1)
            .add(network2)
            .add(network3)
            .add(network4)
            .simplify();

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 4);
        assert_eq!(Some(network1), ip_range.get_network(8, "10.0.0.0"));
        assert_eq!(Some(network2), ip_range.get_network(16, "172.16.0.0"));
        assert_eq!(Some(network3), ip_range.get_network(24, "192.168.1.0"));
        assert_eq!(Some(network4), ip_range.get_network(32, "254.254.254.254"));
    }

    #[test]
    fn merge_disjoint() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "10.0.0.0/8".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        let network3 = "192.168.1.0/24".parse().unwrap();
        let network4 = "254.254.254.254/32".parse().unwrap();
        ip_range1.add(network1).add(network2);
        ip_range2.add(network3).add(network4);

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 4);
        assert_eq!(Some(network1), ip_range.get_network(8, "10.0.0.0"));
        assert_eq!(Some(network2), ip_range.get_network(16, "172.16.0.0"));
        assert_eq!(Some(network3), ip_range.get_network(24, "192.168.1.0"));
        assert_eq!(Some(network4), ip_range.get_network(32, "254.254.254.254"));
    }

    #[test]
    fn merge_joint1() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "172.16.4.0/24".parse().unwrap();
        let network2 = "172.16.4.0/22".parse().unwrap();
        ip_range1.add(network1);
        ip_range2.add(network2);

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(22, "172.16.4.0"));
    }

    #[test]
    fn merge_joint2() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "172.16.5.0/24".parse().unwrap();
        let network2 = "172.16.4.0/22".parse().unwrap();
        ip_range1.add(network1);
        ip_range2.add(network2);

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(22, "172.16.4.0"));
    }

    #[test]
    fn merge_sequent1() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "172.16.4.0/24".parse().unwrap();
        let network2 = "172.16.5.0/24".parse().unwrap();
        let network3 = "172.16.6.0/24".parse().unwrap();
        ip_range1.add(network1);
        ip_range2.add(network2);
        ip_range2.add(network3);

        let ip_range = ip_range1.merge(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 2);
        assert_eq!(
            "172.16.4.0/23".parse().ok(),
            ip_range.get_network(23, "172.16.4.0")
        );
        assert_eq!(
            "172.16.6.0/24".parse().ok(),
            ip_range.get_network(24, "172.16.6.0")
        );
    }

    #[test]
    fn merge_sequent2() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let mut ip_range3 = IpRange::new();
        ip_range1
            .add("192.168.0.0/20".parse().unwrap())
            .add("192.168.24.0/21".parse().unwrap());
        ip_range2
            .add("192.168.16.0/22".parse().unwrap())
            .add("192.168.23.0/24".parse().unwrap());
        ip_range3
            .add("192.168.20.0/24".parse().unwrap())
            .add("192.168.21.0/24".parse().unwrap())
            .add("192.168.22.0/24".parse().unwrap());

        let ip_range = ip_range1.merge(&ip_range2).merge(&ip_range3);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(
            "192.168.0.0/19".parse().ok(),
            ip_range.get_network(19, "192.168.0.0")
        );
    }

    #[test]
    fn intersect_disjoint() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1: Ipv4Net = "10.0.0.0/8".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        let network3 = "192.168.1.0/24".parse().unwrap();
        let network4 = "254.254.254.254/32".parse().unwrap();
        ip_range1.add(network1).add(network2);
        ip_range2.add(network3).add(network4);

        let ip_range = ip_range1.intersect(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 0);
    }

    #[test]
    fn intersect_joint1() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "172.16.4.0/24".parse().unwrap();
        let network2 = "172.16.4.0/22".parse().unwrap();
        ip_range1.add(network1);
        ip_range2.add(network2);

        let ip_range = ip_range1.intersect(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network1), ip_range.get_network(24, "172.16.4.0"));
    }

    #[test]
    fn intersect_joint2() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "172.16.5.0/24".parse().unwrap();
        let network2 = "172.16.4.0/22".parse().unwrap();
        ip_range1.add(network1);
        ip_range2.add(network2);

        let ip_range = ip_range1.intersect(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network1), ip_range.get_network(24, "172.16.5.0"));
    }

    #[test]
    fn intersect_joint3() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "172.16.5.0/24".parse().unwrap();
        let network2 = "172.16.5.0/24".parse().unwrap();
        ip_range1.add(network1);
        ip_range2.add(network2);

        let ip_range = ip_range1.intersect(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network1), ip_range.get_network(24, "172.16.5.0"));
    }

    #[test]
    fn intersect_joint4() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1 = "10.0.0.0/8".parse().unwrap();
        let network2 = "192.168.0.0/24".parse().unwrap();
        let network3 = "10.10.0.0/16".parse().unwrap();
        let network4 = "10.254.0.0/17".parse().unwrap();
        let network5 = "192.168.0.0/16".parse().unwrap();
        ip_range1.add(network1).add(network2);
        ip_range2.add(network3).add(network4).add(network5);

        let ip_range = ip_range1.intersect(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 3);
        assert_eq!(Some(network3), ip_range.get_network(16, "10.10.0.0"));
        assert_eq!(Some(network4), ip_range.get_network(17, "10.254.0.0"));
        assert_eq!(Some(network2), ip_range.get_network(24, "192.168.0.0"));
    }

    #[test]
    fn exclude_disjoint() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1: Ipv4Net = "10.0.0.0/8".parse().unwrap();
        let network2 = "172.16.0.0/16".parse().unwrap();
        let network3 = "192.168.1.0/24".parse().unwrap();
        let network4 = "254.254.254.254/32".parse().unwrap();
        ip_range1.add(network1).add(network2);
        ip_range2.add(network3).add(network4);

        let ip_range = ip_range1.exclude(&ip_range2);
        assert_eq!(ip_range1, ip_range);
    }

    #[test]
    fn exclude_larger() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1: Ipv4Net = "172.16.4.0/24".parse().unwrap();
        let network2 = "192.168.1.0/24".parse().unwrap();
        let network3 = "172.16.4.0/22".parse().unwrap();
        ip_range1.add(network1).add(network2);
        ip_range2.add(network3);

        let ip_range = ip_range1.exclude(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(24, "192.168.1.0"));
    }

    #[test]
    fn exclude_identical() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1: Ipv4Net = "172.16.5.0/24".parse().unwrap();
        let network2 = "192.168.1.0/24".parse().unwrap();
        let network3 = "172.16.4.0/22".parse().unwrap();
        let network4 = "10.0.0.0/8".parse().unwrap();

        ip_range1.add(network1).add(network2);
        ip_range2.add(network3).add(network4);

        let ip_range = ip_range1.exclude(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 1);
        assert_eq!(Some(network2), ip_range.get_network(24, "192.168.1.0"));
    }

    #[test]
    fn exclude_split1() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1: Ipv4Net = "172.16.4.0/22".parse().unwrap();
        let network2 = "192.168.1.0/24".parse().unwrap();
        let network3 = "172.16.5.0/24".parse().unwrap();
        let network4 = "10.0.0.0/8".parse().unwrap();

        ip_range1.add(network1).add(network2);
        ip_range2.add(network3).add(network4);

        let ip_range = ip_range1.exclude(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 3);
        assert_eq!(Some(network2), ip_range.get_network(24, "192.168.1.0"));
        assert_eq!(
            "172.16.4.0/24".parse().ok(),
            ip_range.get_network(24, "172.16.4.0")
        );
        assert_eq!(
            "172.16.6.0/23".parse().ok(),
            ip_range.get_network(23, "172.16.6.0")
        );
    }

    #[test]
    fn exclude_split2() {
        let mut ip_range1 = IpRange::new();
        let mut ip_range2 = IpRange::new();
        let network1: Ipv4Net = "172.16.4.0/22".parse().unwrap();
        let network2 = "192.168.1.0/24".parse().unwrap();
        let network3 = "172.16.4.0/24".parse().unwrap();
        let network4 = "10.0.0.0/8".parse().unwrap();

        ip_range1.add(network1).add(network2);
        ip_range2.add(network3).add(network4);

        let ip_range = ip_range1.exclude(&ip_range2);
        assert_eq!(ip_range.into_iter().count(), 3);
        assert_eq!(Some(network2), ip_range.get_network(24, "192.168.1.0"));
        assert_eq!(
            "172.16.5.0/24".parse().ok(),
            ip_range.get_network(24, "172.16.5.0")
        );
        assert_eq!(
            "172.16.6.0/23".parse().ok(),
            ip_range.get_network(23, "172.16.6.0")
        );
    }
}
