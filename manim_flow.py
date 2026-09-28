"""
3Blue1Brown Manim Animation: BitNet b1.58 Ternary Arithmetic
Illustrates how floating-point matrix multiplication is replaced by integer addition.
"""
from manim import *

class BitNetTernaryFlow(Scene):
    def construct(self):
        title = Text("BitNet b1.58: Ternary Matrix Addition", font_size=32, color=ORANGE)
        subtitle = Text("Eliminating FP16 Multipliers with {-1, 0, +1} Integer Adders", font_size=18, color=LIGHT_GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.2).to_edge(UP)
        self.play(Write(title_group))
        self.wait(1)

        # Traditional GEMM vs BitLinear
        old_gemm = Text("FP16 GEMM: 100% Multiplier Power", font_size=20, color=RED).shift(LEFT * 3)
        new_gemm = Text("BitLinear: Pure Integer Additions", font_size=20, color=GREEN).shift(RIGHT * 3)

        self.play(FadeIn(old_gemm), FadeIn(new_gemm))
        self.wait(1)

        energy = Text("FinOps Impact: 71% DRAM Energy Saved | Zero GPU Dependency", font_size=22, color=TEAL).to_edge(DOWN)
        self.play(FadeIn(energy))
        self.wait(2)
