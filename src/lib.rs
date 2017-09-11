extern crate byteorder;

use std::str::FromStr;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::net::Ipv4Addr;
use std::collections::{BTreeMap, Bound};
use byteorder::{BigEndian, ByteOrder};

pub struct IpRange {
    subnets: BTreeMap<CompactIpv4, CompactIpv4>, // Key: prefix, Value: mask
}

impl IpRange {
    fn new() -> IpRange {
        IpRange {
            subnets: BTreeMap::new(),
        }
    }

    fn push(&mut self, subnet: Subnet) {
        self.subnets.insert(subnet.prefix, subnet.mask);
    }

    fn contains(&self, addr: CompactIpv4) -> bool {
        let result = self.subnets
            .range((Bound::Unbounded, Bound::Included(addr)));
        result
            .last()
            .map_or(false, |(&prefix, &mask)| (addr & mask) == prefix)
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct CompactIpv4(u32);

impl std::ops::BitAnd for CompactIpv4 {
    type Output = CompactIpv4;

    fn bitand(self, other: CompactIpv4) -> CompactIpv4 {
        CompactIpv4(self.0 & other.0)
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

#[derive(Debug, PartialEq, Eq)]
pub struct Subnet {
    prefix: CompactIpv4,
    mask: CompactIpv4,
}

impl FromStr for Subnet {
    type Err = ParseSubnetError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.find('/')
            .ok_or(ParseSubnetError::new("No slash found"))
            .and_then(|slash| {
                let prefix: Result<Ipv4Addr, _> = s[..slash].parse();
                let prefix = prefix
                    .map(|addr| addr.into())
                    .map_err(|_| ParseSubnetError::new("Illegal prefix"));

                let mask: Result<Ipv4Addr, _> = s[(slash + 1)..].parse();
                let mask = mask.map(|addr| addr.into()).or_else(|_| {
                    let size: Result<u32, _> = s[(slash + 1)..].parse();
                    size.map_err(|_| ParseSubnetError::new("Unknown subnet format"))
                        .and_then(|size| if size == 32 {
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
                            mask,
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
                mask: 0b11111111_11111111_11111111_00000000.into(),
            }
        );

        let my_subnet: Subnet = "192.168.5.130/255.255.255.192".parse().unwrap();
        assert_eq!(
            my_subnet,
            Subnet {
                prefix: 0b11000000_10101000_00000101_10000000.into(),
                mask: 0b11111111_11111111_11111111_11000000.into(),
            }
        );

        let my_subnet: Subnet = "10.20.30.40/8".parse().unwrap();
        assert_eq!(
            my_subnet,
            Subnet {
                prefix: 0b00001010_00000000_00000000_00000000.into(),
                mask: 0b11111111_00000000_00000000_00000000.into(),
            }
        );

        let my_subnet: Subnet = "233.233.233.233/32".parse().unwrap();
        assert_eq!(
            my_subnet,
            Subnet {
                prefix: 0b11101001_11101001_11101001_11101001.into(),
                mask: 0b11111111_11111111_11111111_11111111.into(),
            }
        );
    }

    #[test]
    fn contains_ip() {
        let mut ip_range = IpRange::new();
        ip_range.push("192.168.5.130/24".parse().unwrap());

        let ip: Ipv4Addr = "192.168.5.1".parse().unwrap();
        assert!(ip_range.contains(ip.into()))
    }
}
