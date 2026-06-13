from manim import *
import numpy as np


class RotationPreservesAngle(Scene):
    def construct(self):
        self.camera.background_color = "#0D0D0D"

        # Bright colors for visibility on dark background
        Q_COL = "#3FD6A0"      # bright green for Q arrow
        K_COL = "#A89BFF"      # bright purple for K arrow
        ANGLE_COL = "#FFC857"  # bright amber for angle arcs
        BAD_COL = "#FF7A7A"    # soft bright red
        GOOD_COL = "#3FD6A0"   # bright green

        # All text in light off-white for max readability
        TEXT_COL = "#F2F2F2"

        # Geometry
        left_center = np.array([-3.7, -0.4, 0])
        right_center = np.array([3.7, -0.4, 0])
        radius = 1.7

        alpha = 70 * DEGREES
        beta = 25 * DEGREES
        rotation_amt = 55 * DEGREES

        def tip(center, angle, r=radius):
            return center + r * np.array([np.cos(angle), np.sin(angle), 0])

        # ============================================================
        # ACT 1: Title
        # ============================================================

        title = Text(
            "Rotation of Q and K  what changes, what doesn't",
            font_size=32, color=TEXT_COL, weight=BOLD,
        )
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title, shift=UP * 0.1), run_time=0.5)

        # ============================================================
        # ACT 2: Panel headers and circles
        # ============================================================

        left_header = Text(
            "Rotate only Q", font_size=22, color=TEXT_COL, weight=BOLD,
        )
        left_header.move_to(left_center + UP * 2.4)

        right_header = Text(
            "Rotate both Q and K", font_size=22, color=TEXT_COL, weight=BOLD,
        )
        right_header.move_to(right_center + UP * 2.4)

        left_circle = Circle(
            radius=radius, color=TEXT_COL, stroke_width=1.2, stroke_opacity=0.4,
        ).move_to(left_center)
        right_circle = Circle(
            radius=radius, color=TEXT_COL, stroke_width=1.2, stroke_opacity=0.4,
        ).move_to(right_center)

        left_origin = Dot(left_center, radius=0.06, color=TEXT_COL)
        right_origin = Dot(right_center, radius=0.06, color=TEXT_COL)

        divider = DashedLine(
            UP * 2.7, DOWN * 3.2,
            color=TEXT_COL, stroke_width=1, stroke_opacity=0.4,
            dash_length=0.1,
        )

        self.play(
            FadeIn(left_header, shift=UP * 0.05),
            FadeIn(right_header, shift=UP * 0.05),
            run_time=0.4,
        )
        self.play(
            Create(left_circle), Create(right_circle),
            FadeIn(left_origin), FadeIn(right_origin),
            FadeIn(divider),
            run_time=0.5,
        )

        # ============================================================
        # ACT 3: Draw Q and K in both panels
        # ============================================================

        def make_vector(center, angle, color, name):
            arr = Arrow(
                center, tip(center, angle), buff=0, color=color,
                stroke_width=4, max_tip_length_to_length_ratio=0.13,
            )
            lbl = MathTex(name, font_size=32, color=color)
            lbl.move_to(tip(center, angle) + 0.32 * np.array([np.cos(angle), np.sin(angle), 0]))
            return arr, lbl

        lq, lq_lbl = make_vector(left_center, alpha, Q_COL, "Q")
        lk, lk_lbl = make_vector(left_center, beta, K_COL, "K")
        rq, rq_lbl = make_vector(right_center, alpha, Q_COL, "Q")
        rk, rk_lbl = make_vector(right_center, beta, K_COL, "K")

        self.play(
            GrowArrow(lq), FadeIn(lq_lbl),
            GrowArrow(rq), FadeIn(rq_lbl),
            run_time=0.5,
        )
        self.play(
            GrowArrow(lk), FadeIn(lk_lbl),
            GrowArrow(rk), FadeIn(rk_lbl),
            run_time=0.5,
        )

        def make_between_arc(center, a_start, a_end, color=ANGLE_COL, r=0.85):
            return Arc(
                radius=r, start_angle=a_start, angle=(a_end - a_start),
                color=color, stroke_width=3.5, arc_center=center,
            )

        l_arc = make_between_arc(left_center, beta, alpha)
        r_arc = make_between_arc(right_center, beta, alpha)

        mid = (alpha + beta) / 2

        # Angle labels in light text color, not amber (max contrast)
        l_arc_lbl = MathTex(r"\alpha - \beta", font_size=26, color=TEXT_COL)
        l_arc_lbl.move_to(left_center + 1.25 * np.array([np.cos(mid), np.sin(mid), 0]))

        r_arc_lbl = MathTex(r"\alpha - \beta", font_size=26, color=TEXT_COL)
        r_arc_lbl.move_to(right_center + 1.25 * np.array([np.cos(mid), np.sin(mid), 0]))

        self.play(
            Create(l_arc), FadeIn(l_arc_lbl),
            Create(r_arc), FadeIn(r_arc_lbl),
            run_time=0.6,
        )

        # Dot product readouts
        l_dot = MathTex(
            r"Q \cdot K = \cos(\alpha - \beta)",
            font_size=24, color=TEXT_COL,
        )
        l_dot.move_to(left_center + DOWN * 2.4)

        r_dot = MathTex(
            r"Q \cdot K = \cos(\alpha - \beta)",
            font_size=24, color=TEXT_COL,
        )
        r_dot.move_to(right_center + DOWN * 2.4)

        self.play(FadeIn(l_dot), FadeIn(r_dot), run_time=0.5)
        self.wait(0.8)

        # ============================================================
        # ACT 4: Rotate
        # ============================================================

        new_alpha_left = alpha + rotation_amt
        new_alpha_right = alpha + rotation_amt
        new_beta_right = beta + rotation_amt

        def label_target(center, angle):
            return tip(center, angle) + 0.32 * np.array([np.cos(angle), np.sin(angle), 0])

        self.play(
            Rotate(lq, angle=rotation_amt, about_point=left_center),
            lq_lbl.animate.move_to(label_target(left_center, new_alpha_left)),
            FadeOut(l_arc), FadeOut(l_arc_lbl),
            Rotate(rq, angle=rotation_amt, about_point=right_center),
            Rotate(rk, angle=rotation_amt, about_point=right_center),
            rq_lbl.animate.move_to(label_target(right_center, new_alpha_right)),
            rk_lbl.animate.move_to(label_target(right_center, new_beta_right)),
            FadeOut(r_arc), FadeOut(r_arc_lbl),
            run_time=1.6,
        )

        # ============================================================
        # ACT 5: New angle arcs after rotation
        # ============================================================

        new_l_arc = make_between_arc(left_center, beta, new_alpha_left, color=BAD_COL)
        new_l_mid = (new_alpha_left + beta) / 2
        new_l_lbl = MathTex(
            r"(\alpha - \beta) + m\theta",
            font_size=30, color=TEXT_COL,
        )
        new_l_lbl.move_to(left_center + 1.55 * np.array([np.cos(new_l_mid), np.sin(new_l_mid), 0]))

        new_r_arc = make_between_arc(right_center, new_beta_right, new_alpha_right, color=GOOD_COL)
        new_r_mid = (new_alpha_right + new_beta_right) / 2
        new_r_lbl = MathTex(
            r"\alpha - \beta",
            font_size=30, color=TEXT_COL,
        )
        new_r_lbl.move_to(right_center + 1.3 * np.array([np.cos(new_r_mid), np.sin(new_r_mid), 0]))

        self.play(
            Create(new_l_arc), FadeIn(new_l_lbl),
            Create(new_r_arc), FadeIn(new_r_lbl),
            run_time=0.7,
        )

        # Updated dot product readouts — keep text light, color stays via vectors/arcs
        new_l_dot = MathTex(
            r"Q_m \cdot K = \cos\!\big((\alpha-\beta) + m\theta\big)",
            font_size=22, color=TEXT_COL,
        )
        new_l_dot.move_to(l_dot.get_center())

        new_r_dot = MathTex(
            r"Q_m \cdot K_n = \cos(\alpha - \beta)",
            font_size=24, color=TEXT_COL,
        )
        new_r_dot.move_to(r_dot.get_center())

        self.play(
            Transform(l_dot, new_l_dot),
            Transform(r_dot, new_r_dot),
            run_time=0.8,
        )

        # Status tags — keep them colored for emphasis, but bright shades
        l_tag = Text("angle changes", font_size=22, color=BAD_COL, weight=BOLD)
        l_tag.move_to(left_center + DOWN * 3.1)

        r_tag = Text("angle preserved", font_size=22, color=GOOD_COL, weight=BOLD)
        r_tag.move_to(right_center + DOWN * 3.1)

        self.play(
            FadeIn(l_tag, shift=UP * 0.1),
            FadeIn(r_tag, shift=UP * 0.1),
            run_time=0.5,
        )

        self.wait(1.5)

        # ============================================================
        # ACT 6: Caption
        # ============================================================

        self.play(FadeOut(l_tag), FadeOut(r_tag), run_time=0.3)

        caption = Text(
            "Rotating both together preserves the relative angle\nthat carries meaning.",
            font_size=24, color=TEXT_COL, weight=BOLD,
        )
        caption.to_edge(DOWN, buff=0.25)
        self.play(FadeIn(caption, shift=UP * 0.1), run_time=0.7)
        self.wait(2.5)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.6)