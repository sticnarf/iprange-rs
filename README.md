# iprange-rs

[![](https://img.shields.io/crates/v/iprange.svg)](https://crates.io/crates/iprange)

`iprange` is a Rust library for managing IP ranges. 

It provides fast adding and removing operations.

It also provides `merge`, `intersect` and `exclude` methods 
that enable you to manipulate it like a set.

Of course, you can test whether an IP address is in an `IpRange`.

**See the [documentation](https://docs.rs/iprange/) for details.**

## Example

```rust
extern crate iprange;

use std::net::Ipv4Addr;
use iprange::IpRange;

fn main() {
    let mut ip_range = IpRange::new();
    ip_range
        .add("10.0.0.0/8".parse().unwrap())
        .add("172.16.0.0/16".parse().unwrap())
        .add("192.168.1.0/24".parse().unwrap());

    assert!(ip_range.contains("172.16.32.1".parse::<Ipv4Addr>().unwrap()));
    assert!(ip_range.contains("192.168.1.1".parse::<Ipv4Addr>().unwrap()));
}
```

## License

`iprange` is licensed under the MIT license.