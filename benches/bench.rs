#![feature(test)]

extern crate ipnet;
extern crate iprange;
extern crate rand;
extern crate test;

use test::Bencher;
use iprange::*;
use ipnet::Ipv4Net;
use std::net::Ipv4Addr;
use std::fs::File;
use std::path::PathBuf;
use std::io::{BufRead, BufReader};
use rand::{Rng, SeedableRng, StdRng};

#[bench]
fn parse_one_ip(b: &mut Bencher) {
    b.iter(|| "192.168.1.1".parse::<Ipv4Addr>().unwrap());
}

#[bench]
fn parse_one_subnet(b: &mut Bencher) {
    b.iter(|| "192.168.0.0/24".parse::<Ipv4Net>().unwrap());
}

fn chnlists() -> Vec<String> {
    let path = PathBuf::from(file!());
    let f =
        File::open(path.with_file_name("chnlist.txt")).expect("Unable to open chnlist.txt file");
    let reader = BufReader::new(f);
    reader.lines().flat_map(|l| l).collect()
}

fn rand_ip_list(n: usize) -> Vec<Ipv4Addr> {
    let mut rng = StdRng::from_seed(&[1926, 8, 17]);
    (0..n).map(|_| rng.next_u32().into()).collect()
}

#[bench]
fn parse_chnlists(b: &mut Bencher) {
    let lines = chnlists();
    b.iter(|| for line in &lines {
        line.parse::<Ipv4Net>().ok();
    });
}

#[bench]
fn new_ip_range(b: &mut Bencher) {
    b.iter(|| IpRange::<Ipv4Net>::new());
}

#[bench]
fn create_ip_range_with_chnlists(b: &mut Bencher) {
    let chnlists = chnlists();
    b.iter(|| {
        chnlists
            .iter()
            .flat_map(|l| l.parse::<Ipv4Net>())
            .collect::<IpRange<Ipv4Net>>()
    });
}

#[bench]
fn test_10000_ips_in_chnlists(b: &mut Bencher) {
    let ip_list = rand_ip_list(10000);
    let chnlists = chnlists()
        .iter()
        .flat_map(|l| l.parse::<Ipv4Net>())
        .collect::<IpRange<Ipv4Net>>();
    b.iter(|| for ip in &ip_list {
        chnlists.contains(ip);
    });
}
