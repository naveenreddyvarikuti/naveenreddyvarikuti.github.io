from manim import *
import numpy as np


class BinaryDiscontinuity(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        positions = {
            0: [0, 0, 0],
            1: [0, 0, 1],
            2: [0, 1, 0],
            3: [0, 1, 1],
            4: [1, 0, 0],
            5: [1, 0, 1],
            6: [1, 1, 0],
            7: [1, 1, 1],
        }

        dot_colors = {
            0: "#4A90D9",
            1: "#5B9BD5",
            2: "#6CC24A",
            3: "#A8D86E",
            4: "#F5A623",
            5: "#E8743B",
            6: "#D94E7A",
            7: "#C74DCC",
        }

        scale_f = 3.2
        origin = np.array([0, -0.8, 0])

        def project(x, y, z):
            px = scale_f * (x * 0.7 - z * 0.7)
            py = scale_f * (-x * 0.25 - z * 0.25 + y * 0.85)
            return np.array([px, py, 0]) + origin

        # ============================================================
        # ACT 1: Title
        # ============================================================

        title = Text(
            "Binary encodings in 3D space",
            font_size=34, color=WHITE, weight=BOLD,
        )
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title, shift=UP * 0.1), run_time=0.25)

        # ============================================================
        # ACT 2: XYZ axes
        # ============================================================

        axis_len = 1.2

        ax_x = Arrow(
            project(0, 0, 0), project(axis_len, 0, 0),
            color=WHITE, stroke_width=2, buff=0,
            max_tip_length_to_length_ratio=0.06,
        )
        ax_y = Arrow(
            project(0, 0, 0), project(0, axis_len, 0),
            color=WHITE, stroke_width=2, buff=0,
            max_tip_length_to_length_ratio=0.06,
        )
        ax_z = Arrow(
            project(0, 0, 0), project(0, 0, axis_len),
            color=WHITE, stroke_width=2, buff=0,
            max_tip_length_to_length_ratio=0.06,
        )

        lbl_bit2 = Text("bit 2", font_size=18, color=WHITE)
        lbl_bit2.next_to(ax_x.get_end(), RIGHT + DOWN * 0.3, buff=0.1)

        lbl_bit1 = Text("bit 1", font_size=18, color=WHITE)
        lbl_bit1.next_to(ax_y.get_end(), UP + LEFT * 0.2, buff=0.1)

        lbl_bit0 = Text("bit 0", font_size=18, color=WHITE)
        lbl_bit0.next_to(ax_z.get_end(), LEFT + DOWN * 0.3, buff=0.1)

        self.play(Create(ax_x), Create(ax_y), Create(ax_z), run_time=0.25)
        self.play(FadeIn(lbl_bit2), FadeIn(lbl_bit1), FadeIn(lbl_bit0), run_time=0.15)

        # Faint cube edges
        for (a, b) in [
            ((1,0,0),(1,1,0)), ((1,0,0),(1,0,1)),
            ((0,1,0),(1,1,0)), ((0,1,0),(0,1,1)),
            ((0,0,1),(1,0,1)), ((0,0,1),(0,1,1)),
            ((1,1,0),(1,1,1)), ((1,0,1),(1,1,1)), ((0,1,1),(1,1,1)),
        ]:
            line = DashedLine(
                project(*a), project(*b),
                color=WHITE, stroke_width=0.6, stroke_opacity=0.12,
                dash_length=0.05,
            )
            self.add(line)

        # ============================================================
        # ACT 3: Place points with number labels on each dot
        # ============================================================

        dots = {}
        dot_number_labels = {}

        # Offsets for the small number label next to each dot
        # Tuned per point to avoid overlap
        num_offsets = {
            0: DOWN * 0.3,
            1: LEFT * 0.35 + DOWN * 0.1,
            2: RIGHT * 0.35 + DOWN * 0.1,
            3: LEFT * 0.35 + UP * 0.1,
            4: RIGHT * 0.35 + DOWN * 0.1,
            5: RIGHT * 0.35 + UP * 0.1,
            6: LEFT * 0.35,
            7: UP * 0.35,
        }

        for pos_idx in range(8):
            bits = positions[pos_idx]
            coord = project(bits[0], bits[1], bits[2])
            dot = Dot(coord, radius=0.14, color=dot_colors[pos_idx], fill_opacity=1.0)

            num_label = Text(
                str(pos_idx), font_size=18, color=dot_colors[pos_idx], weight=BOLD,
            )
            num_label.move_to(coord + num_offsets[pos_idx])

            dots[pos_idx] = dot
            dot_number_labels[pos_idx] = num_label

            self.play(
                FadeIn(dot, scale=0.5),
                FadeIn(num_label, shift=UP * 0.03),
                run_time=0.08,
            )

        # Legend on the left
        legend_title = Text("Positions", font_size=16, color=WHITE, weight=BOLD)
        legend_entries = VGroup()

        for pos_idx in range(8):
            bits = positions[pos_idx]
            binary_str = ", ".join(str(b) for b in bits)
            color_dot = Dot(radius=0.06, color=dot_colors[pos_idx])
            entry_text = Text(
                f" {pos_idx}  [{binary_str}]",
                font_size=14, color=WHITE,
            )
            row = VGroup(color_dot, entry_text).arrange(RIGHT, buff=0.1)
            legend_entries.add(row)

        legend_entries.arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        legend_box = VGroup(legend_title, legend_entries).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        legend_box.to_corner(UL, buff=0.4).shift(DOWN * 0.6)

        legend_bg = SurroundingRectangle(
            legend_box, color=WHITE, stroke_width=1,
            corner_radius=0.1, buff=0.15, fill_color=BLACK, fill_opacity=0.85,
        )

        self.play(FadeIn(legend_bg), run_time=0.1)
        self.play(FadeIn(legend_title, shift=UP * 0.05), run_time=0.1)
        for entry in legend_entries:
            self.play(FadeIn(entry, shift=UP * 0.03), run_time=0.05)

        self.wait(0.3)

        # ============================================================
        # ACT 4: Connect 0->1->2->3
        # ============================================================

        def make_edge(a, b, color=WHITE, width=2.5):
            pa = project(*positions[a])
            pb = project(*positions[b])
            return Line(pa, pb, color=color, stroke_width=width)

        for (a, b) in [(0, 1), (1, 2), (2, 3)]:
            edge = make_edge(a, b, color=WHITE, width=2)
            self.play(Create(edge), run_time=0.15)

        step_note = Text("1 bit changes per step", font_size=18, color=WHITE)
        step_note.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(step_note, shift=UP * 0.1), run_time=0.2)
        self.wait(0.3)

        # ============================================================
        # ACT 5: The big jump 3 -> 4
        # ============================================================

        self.play(FadeOut(step_note), run_time=0.15)

        ring_3 = Circle(radius=0.24, color="#FF4444", stroke_width=3).move_to(dots[3].get_center())
        ring_4 = Circle(radius=0.24, color="#FF4444", stroke_width=3).move_to(dots[4].get_center())
        self.play(Create(ring_3), Create(ring_4), run_time=0.2)

        big_edge = DashedLine(
            project(*positions[3]), project(*positions[4]),
            color="#FF4444", stroke_width=3.5, dash_length=0.12,
        )
        self.play(Create(big_edge), run_time=0.25)

        # Comparison box
        comp_title = Text("Position 3 vs 4", font_size=20, color=WHITE, weight=BOLD)
        line1 = Text("3:  [0, 1, 1]", font_size=18, color=dot_colors[3])
        line2 = Text("4:  [1, 0, 0]", font_size=18, color=dot_colors[4])
        line3 = Text("      ^   ^   ^", font_size=18, color="#FF4444")
        line4 = Text("3 bits flip!", font_size=20, color="#FF4444", weight=BOLD)

        comp_group = VGroup(comp_title, line1, line2, line3, line4)
        comp_group.arrange(DOWN, buff=0.12, center=True)
        comp_group.to_corner(DR, buff=0.4)

        comp_bg = SurroundingRectangle(
            comp_group, color=WHITE, stroke_width=1.5,
            corner_radius=0.1, buff=0.2, fill_color=BLACK, fill_opacity=0.92,
        )

        self.play(FadeIn(comp_bg), run_time=0.15)
        for item in comp_group:
            self.play(FadeIn(item, shift=UP * 0.05), run_time=0.08)

        self.wait(0.5)

        # ============================================================
        # ACT 6: Continue 4->5->6->7
        # ============================================================

        for (a, b) in [(4, 5), (5, 6), (6, 7)]:
            edge = make_edge(a, b, color=WHITE, width=2)
            self.play(Create(edge), run_time=0.12)

        self.wait(0.3)

        # ============================================================
        # ACT 7: Punchline
        # ============================================================

        punchline = Text(
            "Adjacent positions can be far apart in binary space.",
            font_size=24, color=WHITE,
        )
        punchline.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=0.3)
        self.wait(1.0)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.4)