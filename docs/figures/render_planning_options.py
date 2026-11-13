#!/usr/bin/env python3
"""Render current single-plan selection with the shared EEG-harness primitives."""

from render_figures import Figure, BLUE, INK, LINE, MUTED, PAPER, TEAL, WHITE


def planning_options():
    f = Figure('04-planning-options', 1170, 'Planner and Critic select one research task',
        'Planner uses paper and experiment memory to propose two to four tasks. Critic can retrieve related experimental history before choosing one. New methods go to Designer with selected task context, not the planning history packet. Shown tasks illustrate supported categories, not a required mix.')
    f.header('04', 'Compare plans before writing code.',
             'One Planner proposes alternatives. One Critic compares the complete pool.')
    f.path('M64 210 H724', TEAL, 4)
    f.path('M830 210 H1376', TEAL, 4)
    f.text(64, 256, 'Planner retrieves', 31, TEAL, weight='bold')
    f.text(64, 300, 'Relevant paper and experiment memory', 27, MUTED)
    f.text(830, 256, 'Critic can add', 31, TEAL, weight='bold')
    f.text(830, 300, 'Experiment history related to each plan', 26, MUTED)
    f.text(64, 380, 'Propose 2–4 distinct plans', 34, BLUE, weight='bold')
    rows = [
        ('New method', 'Change a training recipe or EEG encoder.', True),
        ('Confirmation', 'Test a frozen method at full scope.', False),
        ('Validation', 'Evaluate a method in the selected scope.', False),
        ('Diagnosis', 'Analyze existing results without training.', False),
    ]
    for i, (title, detail, selected) in enumerate(rows):
        y = 416 + i * 124
        f.rect(64, y, 660, 108, '#EAF0FF' if selected else WHITE,
               BLUE if selected else LINE, radius=6, width=2 if selected else 1.4)
        f.path(f'M86 {y+22} V{y+86}', BLUE if selected else LINE, 4)
        f.text(108, y+43, title, 31, BLUE if selected else INK, weight='bold')
        f.text(108, y+82, detail, 26, MUTED)
    f.path('M744 470 H764 V842 H744', BLUE, 2)
    f.path('M764 610 H820', BLUE, 2.5, arrow='blue')
    f.rect(830, 416, 546, 320, WHITE, BLUE, radius=8, width=1.6)
    f.text(864, 470, 'Independent Critic', 35, BLUE, weight='bold')
    f.text(864, 521, 'Compare scientific value', 28, INK)
    f.text(864, 565, 'Relate plans to earlier results', 28, INK)
    f.text(864, 609, 'Weigh information and cost', 28, INK)
    f.text(864, 684, 'Choose one original proposal', 27, BLUE, weight='bold')
    f.path('M1103 736 V778', BLUE, 2.5, arrow='blue')
    f.rect(830, 788, 546, 108, BLUE, BLUE, radius=8)
    f.text(864, 832, 'Selected task: new method', 30, WHITE, weight='bold')
    f.text(864, 871, 'An illustrative selection', 26, WHITE)
    f.rule(64, 945, 1376)
    f.text(64, 997, 'For a new method, Designer receives', 31, INK, weight='bold')
    f.text(64, 1043, 'Selected task, this round’s settings, interfaces, and parent source.', 28, MUTED)
    f.text(64, 1085, 'Execution materials are task-specific; planning history stays with Planner and Critic.', 26, MUTED)
    f.footer('Illustrative task types. Each round proposes alternatives allowed by the current study state.')
    f.save()


if __name__ == '__main__':
    planning_options()
    print('Rendered planning options as SVG and a 2160px PNG.')
