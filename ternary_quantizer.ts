/**
 * BitNet b1.58 Ternary Quantization & 2-Bit Weight Packing
 * Converts FP16/FP32 weights into {-1, 0, +1} ternary code words.
 */

export class TernaryQuantizer {
  public static quantizeWeights(weights: number[]): { ternary: Int8Array; scale: number } {
    const absSum = weights.reduce((acc, val) => acc + Math.abs(val), 0);
    const scale = absSum / (weights.length || 1);

    const ternary = new Int8Array(weights.length);
    for (let i = 0; i < weights.length; i++) {
      const normalized = weights[i] / (scale + 1e-7);
      if (normalized > 0.5) {
        ternary[i] = 1;
      } else if (normalized < -0.5) {
        ternary[i] = -1;
      } else {
        ternary[i] = 0;
      }
    }

    return { ternary, scale };
  }
}
