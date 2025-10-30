export function float16ToFloat32(h: number): number {
  // Extract components of float16
  const h_sign: number = h & 0x8000;
  const h_exp: number = h & 0x7c00;
  const h_mant: number = h & 0x03ff;

  // Convert sign bit to float32 position
  const f_sign: number = h_sign << 16;

  if (h_exp === 0x0000) {
    // Subnormal or zero
    if (h_mant === 0) {
      // Zero (positive or negative)
      const buffer: ArrayBuffer = new ArrayBuffer(4);
      const view: Uint32Array = new Uint32Array(buffer);
      view[0] = f_sign;
      return new Float32Array(buffer)[0];
    } else {
      // Subnormal - convert to normalized float32
      let mant: number = h_mant / 1024.0;
      let val: number = mant * Math.pow(2, -14); // equivalent to ldexp(mant, -14)

      // Apply sign
      if (h_sign !== 0) {
        val = -val;
      }
      return val;
    }
  } else if (h_exp === 0x7c00) {
    // Infinity or NaN
    const buffer: ArrayBuffer = new ArrayBuffer(4);
    const view: Uint32Array = new Uint32Array(buffer);
    view[0] = f_sign | 0x7f800000 | (h_mant << 13);
    return new Float32Array(buffer)[0];
  } else {
    // Normal number
    const f_exp: number = ((h_exp >> 10) + 112) << 23;
    const f_mant: number = h_mant << 13;

    const buffer: ArrayBuffer = new ArrayBuffer(4);
    const view: Uint32Array = new Uint32Array(buffer);
    view[0] = f_sign | f_exp | f_mant;
    return new Float32Array(buffer)[0];
  }
}

export function convertFloat16ArrayToFloat32Array(
  uint16Array: Uint16Array
): Float32Array {
  const length: number = uint16Array.length;
  const float32Array: Float32Array = new Float32Array(length);

  // Process multiple items per iteration to reduce loop overhead
  const batchSize = 8;
  let i = 0;

  // Process in batches of 8
  for (; i <= length - batchSize; i += batchSize) {
    for (let j = 0; j < batchSize; j++) {
      float32Array[i + j] = float16ToFloat32(uint16Array[i + j]);
    }
  }

  // Process remaining items
  for (; i < length; i++) {
    float32Array[i] = float16ToFloat32(uint16Array[i]);
  }

  return float32Array;
}
