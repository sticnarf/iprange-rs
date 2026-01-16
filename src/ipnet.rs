use super::{IpNet, IpTrieNode, ToNetwork, TraverseState, MSO_U128, MSO_U32};
use ipnet::{Ipv4Net, Ipv6Net};
use std::net::{Ipv4Addr, Ipv6Addr};

impl IpNet for Ipv4Net {
    type S = Ipv4TraverseState;
    type I = Ipv4PrefixBitIterator;

    #[inline]
    fn prefix_bits(&self) -> Self::I {
        let prefix: u32 = self.addr().into();
        Ipv4PrefixBitIterator {
            prefix,
            prefix_len: self.prefix_len(),
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

#[doc(hidden)]
pub struct Ipv4TraverseState {
    node: *const IpTrieNode,
    prefix: u32,
    prefix_len: u8,
}

impl TraverseState for Ipv4TraverseState {
    type Net = Ipv4Net;

    #[inline]
    fn node(&self) -> *const IpTrieNode {
        self.node
    }

    #[inline]
    fn init(root: &IpTrieNode) -> Self {
        Ipv4TraverseState {
            node: root,
            prefix: 0,
            prefix_len: 0,
        }
    }

    #[inline]
    fn transit(&self, next_node: &IpTrieNode, current_bit: bool) -> Self {
        let mask = if current_bit {
            MSO_U32 >> self.prefix_len
        } else {
            0
        };
        Ipv4TraverseState {
            node: next_node,
            prefix: self.prefix | mask,
            prefix_len: self.prefix_len + 1,
        }
    }

    #[inline]
    fn build(&self) -> Self::Net {
        Ipv4Net::new(self.prefix.into(), self.prefix_len as u8).unwrap()
    }
}

#[doc(hidden)]
pub struct Ipv4PrefixBitIterator {
    prefix: u32,
    prefix_len: u8,
}

impl Iterator for Ipv4PrefixBitIterator {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.prefix_len > 0 {
            let prefix = self.prefix;
            self.prefix <<= 1;
            self.prefix_len -= 1;
            Some(prefix & MSO_U32 != 0)
        } else {
            None
        }
    }
}

impl IpNet for Ipv6Net {
    type S = Ipv6TraverseState;
    type I = Ipv6PrefixBitIterator;

    #[inline]
    fn prefix_bits(&self) -> Self::I {
        Ipv6PrefixBitIterator {
            prefix: self.addr().into(),
            prefix_len: self.prefix_len(),
        }
    }

    #[inline]
    fn prefix_len(&self) -> u8 {
        self.prefix_len()
    }

    #[inline]
    fn with_new_prefix(&self, len: u8) -> Self {
        Ipv6Net::new(self.addr(), len).unwrap().trunc()
    }
}

impl ToNetwork<Ipv6Net> for Ipv6Net {
    #[inline]
    fn to_network(&self) -> Ipv6Net {
        self.trunc()
    }
}

impl ToNetwork<Ipv6Net> for Ipv6Addr {
    #[inline]
    fn to_network(&self) -> Ipv6Net {
        Ipv6Net::new(*self, 128).unwrap()
    }
}

impl ToNetwork<Ipv6Net> for u128 {
    #[inline]
    fn to_network(&self) -> Ipv6Net {
        Ipv6Net::new((*self).into(), 128).unwrap()
    }
}

impl ToNetwork<Ipv6Net> for [u8; 16] {
    #[inline]
    fn to_network(&self) -> Ipv6Net {
        Ipv6Net::new((*self).into(), 128).unwrap()
    }
}

impl ToNetwork<Ipv6Net> for [u16; 8] {
    #[inline]
    fn to_network(&self) -> Ipv6Net {
        Ipv6Net::new((*self).into(), 128).unwrap()
    }
}

#[doc(hidden)]
pub struct Ipv6TraverseState {
    node: *const IpTrieNode,
    prefix: u128,
    prefix_len: u8,
}

impl TraverseState for Ipv6TraverseState {
    type Net = Ipv6Net;

    #[inline]
    fn node(&self) -> *const IpTrieNode {
        self.node
    }

    #[inline]
    fn init(root: &IpTrieNode) -> Self {
        Ipv6TraverseState {
            node: root,
            prefix: 0,
            prefix_len: 0,
        }
    }

    #[inline]
    fn transit(&self, next_node: &IpTrieNode, current_bit: bool) -> Self {
        let mask = if current_bit {
            MSO_U128 >> self.prefix_len
        } else {
            0
        };
        Ipv6TraverseState {
            node: next_node,
            prefix: self.prefix | mask,
            prefix_len: self.prefix_len + 1,
        }
    }

    #[inline]
    fn build(&self) -> Self::Net {
        Ipv6Net::new(self.prefix.into(), self.prefix_len as u8).unwrap()
    }
}

#[doc(hidden)]
pub struct Ipv6PrefixBitIterator {
    prefix: u128,
    prefix_len: u8,
}

impl Iterator for Ipv6PrefixBitIterator {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.prefix_len > 0 {
            let prefix = self.prefix;
            self.prefix <<= 1;
            self.prefix_len -= 1;
            Some(prefix & MSO_U128 != 0)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::IpRange;

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

    impl IpRange<Ipv4Net> {
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

    impl IpRange<Ipv4Net> {
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

        let ip_range = ip_range1.merge(&ip_range2);
        let ip_range = ip_range.merge(&ip_range3);
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

    #[test]
    fn iter_ipv4() {
        let mut data = vec!["1.0.1.0/24", "1.0.2.0/23", "1.0.8.0/21"];
        let ip_range: IpRange<Ipv4Net> = data.iter().map(|net| net.parse().unwrap()).collect();
        let mut nets: Vec<String> = ip_range.iter().map(|net| format!("{}", net)).collect();
        data.sort_unstable();
        nets.sort_unstable();
        assert_eq!(nets, data);
    }

    #[test]
    fn iter_ipv6() {
        let mut data = vec![
            "2400:9a40::/32",
            "2400:9dc0::/32",
            "2400:9e00::/32",
            "2400:a040::/32",
        ];
        let ip_range: IpRange<Ipv6Net> = data.iter().map(|net| net.parse().unwrap()).collect();
        let mut nets: Vec<String> = ip_range.iter().map(|net| format!("{}", net)).collect();
        data.sort_unstable();
        nets.sort_unstable();
        assert_eq!(nets, data);
    }

    #[test]
    fn debug_fmt() {
        let ip_range: IpRange<Ipv4Net> = IpRange::default();
        assert_eq!(format!("{:?}", ip_range), "IpRange []");

        let ip_range: IpRange<Ipv4Net> = ["1.0.1.0/24", "1.0.2.0/23", "1.0.8.0/21"]
            .iter()
            .map(|net| net.parse().unwrap())
            .collect();
        assert_eq!(
            format!("{:?}", ip_range),
            "IpRange [1.0.8.0/21, 1.0.2.0/23, 1.0.1.0/24]"
        );

        let ip_range: IpRange<Ipv4Net> = [
            "192.168.0.0/16",
            "1.0.2.0/23",
            "1.0.8.0/21",
            "127.0.0.0/8",
            "172.16.0.0/12",
        ]
        .iter()
        .map(|net| net.parse().unwrap())
        .collect();
        assert_eq!(
            format!("{:?}", ip_range),
            "IpRange [127.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, ...]"
        );

        let ip_range: IpRange<Ipv6Net> = [
            "2001:4438::/32",
            "2001:4510::/29",
            "2400:1040::/32",
            "2400:12c0::/32",
            "2400:1340::/32",
            "2400:1380::/32",
            "2400:15c0::/32",
        ]
        .iter()
        .map(|net| net.parse().unwrap())
        .collect();
        assert_eq!(
            format!("{:?}", ip_range),
            "IpRange [2001:4510::/29, 2001:4438::/32, 2400:1040::/32, ...]"
        );
    }

    #[test]
    fn remove_all() {
        let mut ip_range = IpRange::new();
        let network: Ipv4Net = "1.0.1.0/24".parse().unwrap();
        ip_range.add(network);
        ip_range.remove(network);
        assert!(ip_range.iter().next().is_none());
    }

    #[test]
    fn add_to_all_zeros() {
        let mut ip_range: IpRange<Ipv4Net> = IpRange::new();
        ip_range.add("0.0.0.0/0".parse().unwrap());
        ip_range.add("127.0.0.1/32".parse().unwrap());
        assert!(ip_range.contains_network("0.0.0.0/0"));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serialize_ipv4_as_binary() {
        let mut ip_range: IpRange<Ipv4Net> = IpRange::new();
        ip_range.add("0.0.0.0/0".parse().unwrap());
        ip_range.add("127.0.0.1/32".parse().unwrap());
        ip_range.add("254.254.254.254/32".parse().unwrap());
        let encoded: Vec<u8> = bincode::serialize(&ip_range).unwrap();
        let decoded_ip_range: IpRange<Ipv4Net> = bincode::deserialize(&encoded[..]).unwrap();
        assert_eq!(ip_range, decoded_ip_range);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serialize_ipv6_as_binary() {
        let mut ip_range: IpRange<Ipv6Net> = IpRange::new();
        ip_range.add("2001:4438::/32".parse().unwrap());
        ip_range.add("2400:1040::/32".parse().unwrap());
        ip_range.add("2400:1340::/32".parse().unwrap());
        let encoded: Vec<u8> = bincode::serialize(&ip_range).unwrap();
        let decoded_ip_range: IpRange<Ipv6Net> = bincode::deserialize(&encoded[..]).unwrap();
        assert_eq!(ip_range, decoded_ip_range);
    }
}
