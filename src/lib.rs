extern crate byteorder;

use std::str::FromStr;
use std::sync::Arc;
use std::cell::RefCell;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::net::Ipv4Addr;
use std::iter::FromIterator;
use std::collections::{btree_map, BTreeMap, Bound};
use byteorder::{BigEndian, ByteOrder};

#[derive(Clone)]
pub struct IpRange {
    subnets: Vec<Arc<RefCell<BTreeMap<CompactIpv4, Subnet>>>>, // Key: prefix
}

impl IpRange {
    pub fn new() -> IpRange {
        IpRange {
            subnets: vec![Arc::new(RefCell::new(BTreeMap::new())); 32],
        }
    }

    pub fn add(&mut self, subnet: Subnet) {
        let prefix_size = subnet.prefix_size as usize;

        for larger in self.subnets[0..prefix_size].iter() {
            if let Some((&larger_prefix, &larger_subnet)) = larger
                .borrow()
                .range((Bound::Unbounded, Bound::Included(subnet.prefix)))
                .next_back()
            {
                if subnet.prefix & larger_subnet.mask() == larger_prefix {
                    return;
                }
            }
        }

        for smaller in self.subnets[prefix_size..32].iter() {
            let to_be_removed: Vec<CompactIpv4> = smaller
                .borrow()
                .range((
                    Bound::Included(subnet.prefix),
                    Bound::Included(subnet.end()),
                ))
                .map(|(&prefix, _)| prefix)
                .collect();
            for prefix in to_be_removed {
                smaller.borrow_mut().remove(&prefix);
            }
        }

        self.subnets[prefix_size]
            .borrow_mut()
            .insert(subnet.prefix, subnet);
    }

    pub fn remove(&mut self, subnet: Subnet) {
        unimplemented!()
        // let outer = self.outer_subnet(subnet);
        // if let Some(outer) = outer {
        //     self.subnets.remove(&outer.prefix);
        //     if outer.prefix < subnet.prefix {
        //         subnet
        //             .prefix
        //             .predecessor()
        //             .map(|pred| self.add_subnet(outer.prefix, !(outer.prefix ^ pred)));
        //     }
        //     if outer.end() > subnet.end() {
        //         subnet
        //             .end()
        //             .successor()
        //             .map(|prefix| self.add_subnet(prefix, !(outer.end() ^ prefix)));
        //     }
        //     return;
        // }

        // let to_be_deleted: Vec<CompactIpv4> = self.subnets
        //     .range((
        //         Bound::Included(subnet.prefix),
        //         Bound::Included(subnet.end()),
        //     ))
        //     .map(|(&prefix, _)| prefix)
        //     .collect();
        // for prefix in to_be_deleted {
        //     self.subnets.remove(&prefix);
        // }
    }

    pub fn merge(&self, other: &IpRange) -> IpRange {
        unimplemented!()
        // self.into_iter().chain(other.into_iter()).collect()
    }

    pub fn intersect(&self, other: &IpRange) -> IpRange {
        unimplemented!()
        // let range1 = self.into_iter().filter(|&subnet| other.includes(subnet));
        // let range2 = other.into_iter().filter(|&subnet| self.includes(subnet));
        // range1.chain(range2).collect()
    }

    pub fn exclude(&self, other: &IpRange) -> IpRange {
        unimplemented!()
        // let mut new = self.clone();
        // for subnet in other.into_iter() {
        //     new.remove(subnet);
        // }
        // new
    }

    pub fn contains<T: Into<CompactIpv4>>(&self, addr: T) -> bool {
        self.find_subnet(addr.into()).is_some()
    }

    pub fn includes(&self, subnet: Subnet) -> bool {
        unimplemented!()
        // self.prev_subnet(subnet)
        //     .map(|prev| prev.includes(subnet))
        //     .unwrap_or_default()
    }

    fn find_subnet(&self, ip: CompactIpv4) -> Option<Subnet> {
        for subnets in self.subnets.iter() {
            let subnets = subnets.borrow();
            if let Some((&prefix, &subnet)) = subnets
                .range((Bound::Unbounded, Bound::Included(ip)))
                .next_back()
            {
                if ip & subnet.mask() == prefix {
                    return Some(subnet);
                }
            }
        }
        None
    }

    fn add_subnet(&mut self, prefix: CompactIpv4, mask: CompactIpv4) {
        // self.subnets.insert(prefix, Subnet { prefix, mask });
    }

    // fn prev_subnet(&self, subnet: Subnet) -> Option<Subnet> {
    //     // self.candidate(subnet.prefix)
    // }

    // fn outer_subnet(&self, subnet: Subnet) -> Option<Subnet> {
    //     self.subnets
    //         .range((Bound::Unbounded, Bound::Included(subnet.prefix)))
    //         .next_back()
    //         .and_then(|(_, &outer)| if outer.includes(subnet) {
    //             Some(outer)
    //         } else {
    //             None
    //         })
    // }

    // fn candidate(&self, addr: CompactIpv4) -> Option<Subnet> {
    //     self.subnets
    //         .range((Bound::Unbounded, Bound::Included(addr)))
    //         .next_back()
    //         .map(|(_, &subnet)| subnet)
    // }
}

impl<'a> FromIterator<Subnet> for IpRange {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Subnet>,
    {
        let mut ip_range = IpRange::new();
        iter.into_iter().fold((), |_, subnet| ip_range.add(subnet));
        ip_range
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct CompactIpv4(u32);

impl std::ops::BitAnd for CompactIpv4 {
    type Output = CompactIpv4;

    fn bitand(self, other: CompactIpv4) -> CompactIpv4 {
        CompactIpv4(self.0 & other.0)
    }
}

impl std::ops::BitOr for CompactIpv4 {
    type Output = CompactIpv4;

    fn bitor(self, other: CompactIpv4) -> CompactIpv4 {
        CompactIpv4(self.0 | other.0)
    }
}

impl std::ops::BitXor for CompactIpv4 {
    type Output = CompactIpv4;

    fn bitxor(self, other: CompactIpv4) -> CompactIpv4 {
        CompactIpv4(self.0 ^ other.0)
    }
}

impl std::ops::Not for CompactIpv4 {
    type Output = CompactIpv4;

    fn not(self) -> CompactIpv4 {
        CompactIpv4(!self.0)
    }
}

impl std::ops::BitAndAssign for CompactIpv4 {
    fn bitand_assign(&mut self, other: CompactIpv4) {
        self.0 &= other.0;
    }
}

impl std::ops::BitOrAssign for CompactIpv4 {
    fn bitor_assign(&mut self, other: CompactIpv4) {
        self.0 |= other.0;
    }
}

impl std::ops::BitXorAssign for CompactIpv4 {
    fn bitxor_assign(&mut self, other: CompactIpv4) {
        self.0 ^= other.0;
    }
}

impl From<Ipv4Addr> for CompactIpv4 {
    fn from(addr: Ipv4Addr) -> CompactIpv4 {
        CompactIpv4(BigEndian::read_u32(&addr.octets()))
    }
}

impl From<u32> for CompactIpv4 {
    fn from(addr: u32) -> CompactIpv4 {
        CompactIpv4(addr)
    }
}

impl Into<u32> for CompactIpv4 {
    fn into(self: CompactIpv4) -> u32 {
        self.0
    }
}

impl fmt::Debug for CompactIpv4 {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut bytes = [0u8; 4];
        BigEndian::write_u32(&mut bytes, self.0);
        write!(f, "{}.{}.{}.{}", bytes[0], bytes[1], bytes[2], bytes[3])
    }
}

impl CompactIpv4 {
    pub fn successor(self) -> Option<CompactIpv4> {
        self.0.checked_add(1).map(|v| CompactIpv4(v))
    }

    pub fn predecessor(self) -> Option<CompactIpv4> {
        self.0.checked_sub(1).map(|v| CompactIpv4(v))
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct Subnet {
    pub prefix: CompactIpv4,
    pub prefix_size: u32,
}

impl Subnet {
    pub fn contains<T: Into<CompactIpv4>>(&self, addr: T) -> bool {
        (addr.into() & self.mask()) == self.prefix
    }

    pub fn end(&self) -> CompactIpv4 {
        self.prefix | !(self.mask())
    }

    pub fn includes(&self, other: Subnet) -> bool {
        self.prefix <= other.prefix && self.end() >= other.end()
    }

    pub fn mask(&self) -> CompactIpv4 {
        CompactIpv4(
            0xffffffffu32
                .checked_shl(32 - self.prefix_size)
                .unwrap_or_default(), // The default value of u32 is 0
        )
    }
}

impl fmt::Debug for Subnet {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:?}/{}", self.prefix, self.prefix_size)
    }
}

impl FromStr for Subnet {
    type Err = ParseSubnetError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.find('/')
            .ok_or(ParseSubnetError::new("No slash found"))
            .and_then(|slash| {
                let prefix = s[..slash]
                    .parse()
                    .map(|addr: Ipv4Addr| addr.into())
                    .map_err(|_| ParseSubnetError::new("Illegal prefix"));

                let mask = s[(slash + 1)..]
                    .parse()
                    .map(|addr: Ipv4Addr| addr.into())
                    .or_else(|_| {
                        s[(slash + 1)..]
                            .parse()
                            .map_err(|_| ParseSubnetError::new("Unknown subnet format"))
                            .and_then(|size: u32| if size == 32 {
                                Ok(0xffffffffu32)
                            } else {
                                0xffffffffu32
                                    .checked_shr(size) // Err when size > 32
                                    .ok_or(ParseSubnetError::new("Prefix size out of range"))
                                    .map(|x| !x)
                            })
                            .map(|mask| CompactIpv4(mask))
                    });

                prefix.and_then(|prefix: CompactIpv4| {
                    mask.map(|mask| {
                        Subnet {
                            prefix: prefix & mask, // Fix subnet
                            prefix_size: (32 - mask.0.trailing_zeros()),
                        }
                    })
                })
            })
    }
}

#[derive(Debug)]
pub struct ParseSubnetError {
    reason: String,
}

impl ParseSubnetError {
    fn new(reason: &str) -> ParseSubnetError {
        ParseSubnetError {
            reason: reason.to_owned(),
        }
    }
}

impl Display for ParseSubnetError {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "ParseSubnetError: {}", self.reason)
    }
}

impl Error for ParseSubnetError {
    fn description(&self) -> &str {
        &self.reason
    }

    fn cause(&self) -> Option<&Error> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_subnet() {
        let my_subnet: Subnet = "192.168.5.130/24".parse().unwrap();
        assert_eq!(
            my_subnet,
            Subnet {
                prefix: 0b11000000_10101000_00000101_00000000.into(),
                prefix_size: 24,
            }
        );

        let my_subnet: Subnet = "192.168.5.130/255.255.255.192".parse().unwrap();
        assert_eq!(
            my_subnet,
            Subnet {
                prefix: 0b11000000_10101000_00000101_10000000.into(),
                prefix_size: 26,
            }
        );

        let my_subnet: Subnet = "10.20.30.40/8".parse().unwrap();
        assert_eq!(
            my_subnet,
            Subnet {
                prefix: 0b00001010_00000000_00000000_00000000.into(),
                prefix_size: 8,
            }
        );

        let my_subnet: Subnet = "233.233.233.233/32".parse().unwrap();
        assert_eq!(
            my_subnet,
            Subnet {
                prefix: 0b11101001_11101001_11101001_11101001.into(),
                prefix_size: 32,
            }
        );
    }

    #[test]
    fn with_single_subnet() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.5.130/24".parse().unwrap());
        let ip: Ipv4Addr = "192.168.5.1".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.6.1".parse().unwrap();
        assert!(!ip_range.contains(ip));
    }

    #[test]
    fn with_multiple_subnets() {
        let mut ip_range = IpRange::new();
        ip_range.add("172.16.5.130/24".parse().unwrap());
        let ip: Ipv4Addr = "172.16.6.1".parse().unwrap();
        assert!(!ip_range.contains(ip));
        ip_range.add("172.16.3.1/16".parse().unwrap());
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "172.16.31.1".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "172.17.0.1".parse().unwrap();
        assert!(!ip_range.contains(ip));

        let mut ip_range = IpRange::new();
        ip_range.add("192.168.5.1/24".parse().unwrap());
        ip_range.add("192.168.6.1/22".parse().unwrap());
        ip_range.add("192.168.128.1/20".parse().unwrap());
        let ip: Ipv4Addr = "192.168.1.1".parse().unwrap();
        assert!(!ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.4.1".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.130.1".parse().unwrap();
        assert!(ip_range.contains(ip));

        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/24".parse().unwrap());
        ip_range.add("192.168.1.0/24".parse().unwrap());
        let ip: Ipv4Addr = "192.168.0.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.1.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.2.252".parse().unwrap();
        assert!(!ip_range.contains(ip));
        ip_range.add("192.168.0.0/18".parse().unwrap());
        assert!(!ip_range.contains(ip));

        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/22".parse().unwrap());
        ip_range.add("192.168.4.0/24".parse().unwrap());
        let ip: Ipv4Addr = "192.168.0.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.2.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.4.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.5.252".parse().unwrap();
        assert!(!ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.6.252".parse().unwrap();
        assert!(!ip_range.contains(ip));
    }

    #[test]
    fn remove_subnets() {
        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/22".parse().unwrap());
        ip_range.remove("192.168.2.0/23".parse().unwrap());
        let ip: Ipv4Addr = "192.168.0.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.1.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.2.252".parse().unwrap();
        assert!(!ip_range.contains(ip));

        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/22".parse().unwrap());
        ip_range.remove("192.168.0.0/23".parse().unwrap());
        let ip: Ipv4Addr = "192.168.2.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.3.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.1.252".parse().unwrap();
        assert!(!ip_range.contains(ip));

        let mut ip_range = IpRange::new();
        ip_range.add("192.168.0.0/16".parse().unwrap());
        ip_range.remove("192.168.128.0/24".parse().unwrap());
        let ip: Ipv4Addr = "192.168.127.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.1.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.129.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.252.252".parse().unwrap();
        assert!(ip_range.contains(ip));
        let ip: Ipv4Addr = "192.168.128.252".parse().unwrap();
        assert!(!ip_range.contains(ip));
    }
}
