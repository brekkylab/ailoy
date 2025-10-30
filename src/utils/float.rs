pub mod float16 {
    pub fn f16_to_f32(val: u16) -> f32 {
        let sign = (val >> 15) & 0x1;
        let exponent = (val >> 10) & 0x1f;
        let fraction = val & 0x3ff;

        let result_bits: u32 = if exponent == 0x1f {
            // Infinity or NaN
            if fraction == 0 {
                // Infinity
                (sign as u32) << 31 | 0x7f800000
            } else {
                // NaN
                (sign as u32) << 31 | 0x7f800000 | ((fraction as u32) << 13)
            }
        } else if exponent == 0 {
            // Zero or Subnormal
            if fraction == 0 {
                // Zero
                (sign as u32) << 31
            } else {
                // Subnormal: Denormalized f16 to normalized f32
                let mut exponent_f32 = 127 - 14; // f16 min_exp_bias - f32_bias + 1
                let mut fraction_f32 = fraction as u32;

                // Normalize the subnormal number
                while (fraction_f32 & 0x400) == 0 {
                    fraction_f32 <<= 1;
                    exponent_f32 -= 1;
                }
                fraction_f32 &= !0x400; // Remove the implicit leading 1

                (sign as u32) << 31 | (exponent_f32 << 23) | (fraction_f32 << 13)
            }
        } else {
            // Normalized number
            let f32_exponent = (exponent as u32) + (127 - 15);
            let f32_fraction = (fraction as u32) << 13;

            (sign as u32) << 31 | (f32_exponent << 23) | f32_fraction
        };

        f32::from_bits(result_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion() {
        assert_eq!(float16::f16_to_f32(0x3c00), 1.0); // 1.0
        assert_eq!(float16::f16_to_f32(0xc000), -2.0); // -2.0
        assert!(float16::f16_to_f32(0x7c00).is_infinite()); // +inf
        assert!(float16::f16_to_f32(0xfc00).is_infinite()); // -inf
        assert!(float16::f16_to_f32(0x7e00).is_nan()); // NaN
    }

    #[test]
    fn test_f16_to_f32_all_u16_values() {
        let mut panic_inputs = Vec::new();

        for i in 0u16..=65535 {
            let result = std::panic::catch_unwind(|| {
                float16::f16_to_f32(i);
            });

            if result.is_err() {
                panic_inputs.push(i);
            }
        }

        if !panic_inputs.is_empty() {
            println!("Panicked inputs:");
            println!("{:?}", panic_inputs);
            panic!("Test failed: Panicked for some inputs.");
        } else {
            println!("Test passed: No panics occurred for any input.");
        }
    }
}
