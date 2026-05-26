from manim import *
import numpy as np


class BagOfWordsProblem(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        # Realistic softmax attention scores for "Dog bites man"
        # Each row sums to ~1.0
        # "Dog" attends most to "bites" (subject verb)
        # "bites" distributes roughly evenly
        # "man" attends most to "bites" (object verb)
        vals = [
            [0.34, 0.41, 0.25],
            [0.30, 0.38, 0.32],
            [0.22, 0.44, 0.34],
        ]

        def build_matrix(row_labels, col_labels, title_str, values):
            cs = 0.75
            group = VGroup()

            t = Text(title_str, color=WHITE, font_size=18, weight=BOLD)

            cells = VGroup()
            val_texts = VGroup()
            for r in range(3):
                for c in range(3):
                    sq = Square(
                        side_length=cs,
                        stroke_color=WHITE,
                        stroke_width=1,
                    )
                    intensity = values[r][c]
                    sq.set_fill(WHITE, opacity=intensity * 0.65)
                    sq.move_to(np.array([c * cs, -r * cs, 0]))

                    vt = Text(
                        f"{values[r][c]:.2f}",
                        color=WHITE if intensity < 0.35 else BLACK,
                        font_size=13,
                    ).move_to(sq.get_center())

                    cells.add(sq)
                    val_texts.add(vt)

            grid = VGroup(cells, val_texts)
            grid.move_to(ORIGIN)

            rl = VGroup()
            for i, label in enumerate(row_labels):
                txt = Text(label, color=WHITE, font_size=14)
                txt.next_to(cells[i * 3], LEFT, buff=0.15)
                rl.add(txt)

            cl = VGroup()
            for i, label in enumerate(col_labels):
                txt = Text(label, color=WHITE, font_size=14)
                txt.next_to(cells[i], UP, buff=0.15)
                cl.add(txt)

            t.next_to(cl, UP, buff=0.25)

            group.add(t, grid, rl, cl)
            return group

        # Sentence B: reorder rows and cols
        # "Man bites dog" -> row/col order is man, bites, dog
        reorder = [2, 1, 0]
        vals_b = [
            [vals[reorder[r]][reorder[c]] for c in range(3)]
            for r in range(3)
        ]

        mat_a = build_matrix(
            ["Dog", "bites", "man"],
            ["Dog", "bites", "man"],
            '"Dog bites man"',
            vals,
        )

        mat_b = build_matrix(
            ["Man", "bites", "dog"],
            ["Man", "bites", "dog"],
            '"Man bites dog"',
            vals_b,
        )

        mats = VGroup(mat_a, mat_b).arrange(RIGHT, buff=2.2)
        mats.move_to(UP * 0.5)

        # Animate matrix A: labels first, then cells row by row
        self.play(
            FadeIn(mat_a[0], shift=UP * 0.1),
            FadeIn(mat_a[2], shift=LEFT * 0.1),
            FadeIn(mat_a[3], shift=DOWN * 0.1),
            run_time=0.5,
        )
        for r in range(3):
            row_cells = VGroup(*[mat_a[1][0][r * 3 + c] for c in range(3)])
            row_vals = VGroup(*[mat_a[1][1][r * 3 + c] for c in range(3)])
            self.play(
                FadeIn(row_cells, shift=UP * 0.05),
                FadeIn(row_vals, shift=UP * 0.05),
                run_time=0.3,
            )

        self.wait(0.4)

        # Animate matrix B
        self.play(
            FadeIn(mat_b[0], shift=UP * 0.1),
            FadeIn(mat_b[2], shift=RIGHT * 0.1),
            FadeIn(mat_b[3], shift=DOWN * 0.1),
            run_time=0.5,
        )
        for r in range(3):
            row_cells = VGroup(*[mat_b[1][0][r * 3 + c] for c in range(3)])
            row_vals = VGroup(*[mat_b[1][1][r * 3 + c] for c in range(3)])
            self.play(
                FadeIn(row_cells, shift=UP * 0.05),
                FadeIn(row_vals, shift=UP * 0.05),
                run_time=0.3,
            )

        self.wait(0.6)

        # Highlight same pair
        line1 = Text(
            "score(dog, bites) = 0.41 in both matrices",
            color=WHITE,
            font_size=17,
        ).next_to(mats, DOWN, buff=0.6)

        self.play(FadeIn(line1, shift=UP * 0.1), run_time=0.5)
        self.wait(0.5)

        line2 = Text(
            "Every token pair produces the same score regardless of position.",
            color=WHITE,
            font_size=17,
        ).next_to(line1, DOWN, buff=0.25)

        self.play(FadeIn(line2, shift=UP * 0.1), run_time=0.5)
        self.wait(1)

        # Box around both matrices with IDENTICAL label
        box = SurroundingRectangle(
            mats, color=WHITE, buff=0.25, stroke_width=1.5,
            corner_radius=0.1,
        )
        identical = Text(
            "IDENTICAL",
            color=WHITE,
            font_size=22,
            weight=BOLD,
        ).next_to(box, UP, buff=0.15)

        self.play(
            Create(box),
            FadeIn(identical, shift=DOWN * 0.1),
            run_time=0.7,
        )
        self.wait(3)

        # Fade everything
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=0.8,
        )


