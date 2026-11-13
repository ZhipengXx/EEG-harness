#!/usr/bin/env python3
"""Render EEG-harness's editable, data-free README illustrations.

Run from any directory with Python 3 and Inkscape installed. No model, API,
training data, or third-party Python package is required.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PAPER = "#F5F8FC"
INK = "#15324F"
MUTED = "#5B6B7C"
LINE = "#C9D6E4"
TEAL = "#008478"
BLUE = "#315BE8"
WHITE = "#FFFFFF"
SANS = "DejaVu Sans"


class Figure:
    def __init__(self, name: str, height: int, title: str, description: str):
        self.name, self.height = name, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="{height}" '
            f'viewBox="0 0 1440 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{html.escape(title)}</title>',
            f'<desc id="desc">{html.escape(description)}</desc>',
            '<defs>',
            *[
                f'<marker id="arrow-{name}" markerWidth="9" markerHeight="9" refX="7" '
                f'refY="4" orient="auto" markerUnits="userSpaceOnUse">'
                f'<path d="M0 0 L8 4 L0 8" fill="{color}"/></marker>'
                for name, color in (("ink", INK), ("teal", TEAL), ("blue", BLUE))
            ],
            '</defs>',
            f'<rect width="1440" height="{height}" fill="{PAPER}"/>',
        ]

    def text(self, x, y, label, size=24, color=INK, font=SANS, weight="normal", anchor="start", spacing=0):
        self.parts.append(
            f'<text x="{x}" y="{y}" fill="{color}" font-family="{font}" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
            f'letter-spacing="{spacing}">{html.escape(label)}</text>'
        )

    def rect(self, x, y, w, h, fill=WHITE, stroke=LINE, radius=0, width=1.2):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def path(self, d, color=INK, width=2, fill="none", dash=False, arrow=None, opacity=1):
        self.parts.append(
            f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"'
            + (' stroke-dasharray="7 7"' if dash else '')
            + (f' marker-end="url(#arrow-{arrow})"' if arrow else '') + '/>'
        )

    def circle(self, x, y, r, fill=TEAL, stroke="none", width=1):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def rule(self, x1, y, x2, color=LINE):
        self.path(f'M{x1} {y} H{x2}', color, 1)

    def header(self, index, title, subtitle):
        self.text(64, 49, "EEG-harness", 24, BLUE, weight="bold")
        self.text(64, 114, title, 48, weight="bold")
        self.text(64, 163, subtitle, 27, MUTED)

    def footer(self, text):
        self.rule(64, self.height - 70, 1376)
        self.text(64, self.height - 31, text, 24, MUTED)

    def save(self):
        path = ROOT / f'{self.name}.svg'
        path.write_text('\n'.join(self.parts + ['</svg>']) + '\n', encoding='utf-8')
        subprocess.run([
            'inkscape', str(path), '--export-type=png', '--export-width=2160',
            f'--export-filename={ROOT / (self.name + ".png")}',
        ], check=True, capture_output=True)


def brain(f, x, y, scale=1, electrodes=False, activity=False):
    """A stylized lateral hemisphere; no anatomical or measured-data claim."""
    f.parts.append(f'<g transform="translate({x} {y}) scale({scale})">')
    f.path('M22 111 C2 90 8 54 30 43 C32 20 57 10 78 17 C96 0 126 5 139 18 '
           'C165 10 190 24 193 47 C220 58 224 90 209 107 C211 134 182 151 157 146 '
           'L136 151 C117 155 108 138 111 124 C90 132 71 125 57 127 C39 130 27 122 22 111 Z',
           INK, 2.6, '#E6EFF7')
    folds = [
        'M32 48 C58 31 73 47 71 67 C67 84 43 73 43 94 C43 109 62 114 77 103',
        'M80 22 C97 43 89 54 103 67 C112 77 125 67 126 54 C128 36 145 30 162 39',
        'M45 38 C48 24 65 22 77 29',
        'M111 21 C123 31 113 46 106 49',
        'M177 47 C158 49 149 65 161 78 C169 87 193 74 202 91',
        'M82 76 C100 73 110 88 99 102 C91 114 75 115 79 122',
        'M116 94 C135 83 147 89 146 106 C145 126 167 132 188 117',
        'M164 89 C181 88 191 101 184 111',
        'M118 124 C131 116 138 131 140 142',
    ]
    for d in folds:
        f.path(d, TEAL, 1.7, opacity=.65)
    if activity:
        for cx, cy, r in ((40, 87, 10), (59, 100, 7), (167, 59, 8), (180, 104, 11)):
            f.circle(cx, cy, r, BLUE, width=0)
    if electrodes:
        for cx, cy in ((34, 36), (73, 12), (113, 10), (161, 22), (199, 57)):
            f.circle(cx, cy, 6, PAPER, BLUE, 2)
    f.parts.append('</g>')


def traces(f, x, y, w=250, rows=3, amp=12):
    for row in range(rows):
        baseline = y + row * 32
        f.path(f'M{x} {baseline} H{x+w}', LINE, .8)
        points = []
        for k in range(121):
            v = (math.sin(k * .28 + row * .8) * .46 + math.sin(k * .71 + row) * .22
                 + math.sin(k * .13 + row) * .3)
            points.append(f'{x+k*w/120:.2f} {baseline-amp*v:.2f}')
        f.path('M' + ' L'.join(points), TEAL if row % 2 == 0 else BLUE, 1.9)


def embedding(f, x, y, cell=12):
    for row in range(5):
        for col in range(8):
            f.rect(x + col * (cell+4), y + row * (cell+4), cell, cell,
                   (TEAL, '#9DD6CF', '#DAECEA', BLUE, '#BFCCFA')[(row*3+col*2) % 5], 'none')


def gallery(f, x, y, width=63, rows=1, highlight=True):
    for row in range(rows):
        for col in range(3):
            xx, yy = x+col*(width+14), y+row*(width+14)
            color = BLUE if highlight and col == 1 and row == 0 else LINE
            f.rect(xx, yy, width, width, '#E8EFF7', color, width=2 if color == BLUE else 1)
            f.circle(xx+width*.72, yy+width*.23, width*.08, '#9FB5F5')
            f.path(f'M{xx+6} {yy+width-8} L{xx+width*.32} {yy+width*.45} '
                   f'L{xx+width*.52} {yy+width*.65} L{xx+width*.72} {yy+width*.42} '
                   f'L{xx+width-6} {yy+width-8} Z', 'none', 0, '#8CBBB9')



def hero():
    f = Figure('00-brain-decoding-hero', 650,
        'EEG-harness: from a research question to the next experiment',
        'An agent research harness for EEG and MEG image retrieval. The neural signal, brain and gallery are conceptual illustrations, not experiment results.')
    f.text(64, 73, 'Agentic research for brain decoding', 28, BLUE)
    f.text(60, 213, 'EEG-harness', 100, weight='bold')
    f.text(64, 291, 'From a research question', 42, weight='bold')
    f.text(64, 345, 'to the next experiment.', 42, weight='bold')
    f.text(64, 407, 'Plan, implement, measure, and learn.', 29, MUTED)
    brain(f, 1070, 135, 1.16, electrodes=True)
    traces(f, 1000, 355, 336, rows=3, amp=14)
    f.rule(64, 478, 1376)
    f.text(64, 530, 'EEG / MEG image retrieval', 30, weight='bold')
    f.text(780, 530, 'Literature + experiment memory', 30, TEAL, weight='bold')
    f.footer('Research workflow illustration. Signals and activity colors are schematic.')
    f.save()


def overview():
    f = Figure('01-system-overview', 800, 'Two workflows, one workbench',
        'EEG/MEG retrieval research and fMRI response screening share a workbench, with separate protocols, state and memory.')
    f.header('01', 'Two workflows, one workbench.', 'Start from the scientific question and the data you have.')
    f.path('M64 209 H668', BLUE, 5)
    f.path('M776 209 H1376', TEAL, 5)
    f.text(64, 266, 'EEG / MEG retrieval', 36, weight='bold')
    f.text(776, 266, 'fMRI screening', 36, weight='bold')
    traces(f, 80, 325, 238, rows=3, amp=14)
    f.path('M350 359 H401', BLUE, 2.4, arrow='blue')
    gallery(f, 432, 315, 65)
    brain(f, 800, 295, .78, activity=True)
    f.path('M1008 357 H1059', TEAL, 2.4, arrow='teal')
    for row in range(5):
        for col in range(7):
            f.rect(1094+col*29, 310+row*25, 24, 20,
                   '#C9E7E3' if (row+col)%3 else '#9FB5F5', 'none')
    for y, line in ((478, 'Preprocessed signals and image features'),
                    (527, 'Design and implement a candidate.'),
                    (570, 'Train, compare, and plan the next round.')):
        f.text(64, y, line, 26, MUTED)
    for y, line in ((478, 'Existing series or a configured TRIBE backend'),
                    (527, 'Choose checks for predicted responses.'),
                    (570, 'Measure, inspect, and write a scoped report.')):
        f.text(776, y, line, 25, MUTED)
    f.text(64, 654, 'Code, comparisons, and memory', 27, BLUE, weight='bold')
    f.text(776, 654, 'Screening evidence and diagnostics', 27, TEAL, weight='bold')
    f.footer('The two workflows keep separate protocols, state, and memory.')
    f.save()


def node(f, x, y, width, title, lines, *, color=BLUE, height=154):
    f.rect(x, y, width, height, WHITE, LINE, radius=8, width=1.4)
    f.path(f'M{x+22} {y+23} V{y+height-23}', color, 4)
    f.text(x+42, y+49, title, 31, color, weight='bold')
    for index, line in enumerate(lines):
        f.text(x+42, y+90+index*34, line, 25, MUTED)


def collaboration():
    f = Figure('02-eeg-agent-collaboration', 1380, 'The EEG-harness research loop',
        'Paper and experiment memory inform Planner and Critic. A new method proceeds through design, coding and review, GPU measurement, analysis and local Observation publication. Confirmation and validation reuse code; diagnosis uses existing evidence.')
    f.header('02', 'Each experiment informs the next.',
             'The new-method path, with separate routes for reuse and diagnosis.')
    node(f, 64, 202, 600, 'Paper memory',
         ['Findings and reported setups', 'Separate retrieval allowance'], color=TEAL, height=144)
    node(f, 776, 202, 600, 'Experiment memory',
         ['Measured outcomes and analysis', 'Current and relevant history'], color=TEAL, height=144)
    # Both memory stores inform the Planner; Critic can add related experiment history.
    f.path('M364 346 V377 H344 V414', TEAL, 2, arrow='teal')
    f.path('M1028 346 V377 H364', TEAL, 2)
    f.path('M1200 346 V414', TEAL, 2, arrow='teal')
    node(f, 64, 414, 560, 'Planner',
         ['Propose 2–4 distinct plans', 'Use retrieved literature and results'], height=162)
    node(f, 816, 414, 560, 'Critic',
         ['Compare the complete proposal pool', 'Add related experiment history'], height=162)
    f.path('M624 495 H806', BLUE, 2.5, arrow='blue')
    f.text(720, 452, 'plans +', 24, MUTED, anchor='middle')
    f.text(720, 480, 'evidence', 24, MUTED, anchor='middle')
    f.text(64, 640, 'Selected new method', 30, BLUE, weight='bold')
    f.path('M1096 576 V665 H254 V692', BLUE, 2.5, arrow='blue')
    node(f, 64, 702, 380, 'Designer', ['Specify this experiment', 'Use task-bound context'])
    node(f, 532, 702, 380, 'Coder + Reviewer', ['Implement and review', 'Repair when needed'])
    node(f, 1000, 702, 376, 'GPU experiment', ['Train and evaluate', 'Record measurements'], color=TEAL)
    f.path('M444 779 H522', BLUE, 2.5, arrow='blue')
    f.path('M912 779 H990', BLUE, 2.5, arrow='blue')
    node(f, 64, 924, 380, 'Observation', ['Save facts and analysis', 'Update stored memory'], color=TEAL)
    node(f, 532, 924, 380, 'Analysis', ['Interpret the results', 'Limits and next steps'])
    f.path('M1188 856 V1001 H922', TEAL, 2.5, arrow='teal')
    f.text(1212, 926, 'measured', 24, TEAL)
    f.text(1212, 959, 'evidence', 24, TEAL)
    f.path('M532 1001 H454', TEAL, 2.5, arrow='teal')
    f.path('M64 1001 H28 V495 H54', TEAL, 2, dash=True, arrow='teal')
    f.text(64, 1136, 'Confirmation / validation: reuse code → measure → analyze.', 28, INK)
    f.text(64, 1182, 'Diagnosis: inspect existing evidence → analyze.', 28, INK)
    f.text(64, 1245, 'Auditor / Curator are optional paths; paid maintenance defaults to off.', 26, MUTED)
    f.footer('Models interpret and propose. Local code binds measurements and saves observations.')
    f.save()


def screening():
    f = Figure('03-fmri-screening-loop', 970, 'The fMRI screening workflow',
        'Screen an existing surface series or an image-generated TRIBE response with a gray control. Select checks, measure responses and update evidence. Reports stay within numeric consistency scope.')
    f.header('03', 'Inspect a predicted brain response.',
             'A separate workflow for selecting checks and recording screening evidence.')
    f.rect(64, 231, 360, 510, WHITE, LINE, radius=8)
    f.text(96, 288, 'Input', 36, TEAL, weight='bold')
    brain(f, 137, 326, .9, activity=True)
    f.text(96, 529, 'Surface series', 28, weight='bold')
    f.text(96, 576, 'or image → TRIBE', 26, MUTED)
    f.text(96, 615, 'with a gray control', 26, MUTED)
    f.text(96, 696, 'Run the initial checks.', 25, TEAL)
    node(f, 510, 231, 400, 'Choose checks',
         ['Rule policy or LLM', 'Use the current evidence'], height=204)
    node(f, 980, 231, 396, 'Measure',
         ['Run selected tools', 'Record their outputs'], color=TEAL, height=204)
    node(f, 980, 537, 396, 'Update evidence',
         ['Inspect measurements', 'Continue or finish'], color=TEAL, height=204)
    node(f, 510, 537, 400, 'Write report',
         ['Record screening outcome', 'Scope and stopping reason'], height=204)
    f.path('M424 333 H500', TEAL, 2.4, arrow='teal')
    f.path('M910 333 H970', BLUE, 2.4, arrow='blue')
    f.path('M1178 435 V527', TEAL, 2.4, arrow='teal')
    f.path('M980 639 H920', TEAL, 2.4, arrow='teal')
    f.path('M1376 639 H1410 V483 H710 V445', TEAL, 2, dash=True, arrow='teal')
    f.text(990, 474, 'Replan as needed', 24, TEAL, anchor='middle')
    f.text(64, 819, 'Numeric, temporal, surface, ROI, reference, and specificity checks', 28, INK)
    f.text(64, 864, 'Report scope: numeric consistency only.', 28, TEAL, weight='bold')
    f.footer('Optional semantic and CortexMAE tools depend on configuration and available resources.')
    f.save()


if __name__ == '__main__':
    hero()
    overview()
    collaboration()
    screening()
    print('Rendered four SVG sources and four 2160px PNG exports.')
