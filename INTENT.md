Quick question. Philnits gives out its exam questions on their website and I have a couple of their sheets downloaded. I have Claude Code and can probably analyze each of them and get patterns and whatnot. What can I do from here? I wanna go top down. General ideas (stuff I'm comfortable with), then work my way down the stuff that requires a lot of cognition/reason (questions involving dsa, stuff that requires you to trace and debug - and even code), then the edge cases, in terms of knowledge and in terms of what the Philippine curriculum might not teach you - as well as things that are niche trivia.

Intuition above all. Speed and recall. I my brain is not a library, it's an amorphous blob that takes the form it needs to at any given moment.

Oh hell yes. This is exactly the right instinct.
You’re basically asking: “How do I distill PhilNITS into a controllable phase space?”
Top-down, intuition-first, then harden the sharp edges. Let’s do that.

I’ll give you a concrete, Claude-Code–friendly pipeline, not generic advice.

Big Picture: What you’re really building

You’re not analyzing questions.
You’re building a decision engine with three layers:

Fast recognition layer (90% of questions)

Deliberate reasoning layer (DSA / tracing / debugging)

Exception handling layer (curriculum gaps + trivia)

Claude Code is perfect for this because it’s good at clustering, patterning, and abstraction—which mirrors how you think.

Phase 0 — Normalize the data (don’t skip this)

Before analysis, do one mechanical pass:

Convert all PDFs → clean text

Split into:

question

choices

correct answer

explanation (if present)

Assign each question a unique ID

This lets you treat questions as data, not study material.

Your goal is to stop seeing “questions” and start seeing instances of decision classes.

Phase 1 — Top-down clustering (this is the money step)

Use Claude Code to do unsupervised clustering of questions.

Prompt it roughly like:

“Cluster these questions by what kind of thinking they require, not by topic.”

You’re looking for clusters like:

definition recall

invariant reasoning

control-flow tracing

asymptotic comparison

boundary-condition traps

architecture tradeoffs

operational best practice

⚠️ Important:
Do not let it cluster by labels like “OS”, “DB”, “Network”.
Those are surface features. You want cognitive moves.

Output you want:

A list like:

Cluster A: “Single-step factual judgment”

Cluster B: “State evolution over time”

Cluster C: “Resource tradeoff under constraints”

Cluster D: “Algorithm behavior under input mutation”

Cluster E: “Policy / best-practice alignment”

This becomes your mental map.

Phase 2 — Rank clusters by cognitive cost

Now tell Claude:

“For each cluster, estimate:

average time to solve

failure rate for untrained examinees

penalty for overthinking”

This gives you a priority ordering that aligns with speed + intuition.

You’ll usually see:

High volume, low effort clusters → free points

Medium volume, medium effort → train recognition

Low volume, high effort → deliberate drills only

This is how you avoid burning energy on low ROI prep.

Phase 3 — Build “default instincts” per cluster

For each cluster, explicitly extract:

What the exam usually wants

What it is not rewarding

The boring, professional answer

Example output (you want things like this):

“In architecture tradeoff questions:

Prefer simplicity

Avoid premature optimization

Choose maintainability unless constraints say otherwise”

These become preloaded reflexes.

You’re turning intuition into compiled heuristics.

Phase 4 — Deep work lane: DSA / tracing / debugging

Now we go down the stack.

Sub-cluster these questions further into:

array / pointer tracing

recursion + stack

graph traversal

time/space reasoning

off-by-one and termination bugs

Then for each sub-cluster:

Ask Claude:

“What is the minimal mental simulation needed to solve this correctly?”

You’re hunting for:

invariants

loop summaries

state compression tricks

Example:

Instead of tracing line-by-line, track:

pointer movement pattern

monotonicity

termination condition

This matches your “amorphous blob” brain:
track shape, not steps.

Phase 5 — Edge cases & curriculum gaps (Philippines-specific)

This is where Claude shines as a contrast engine.

Ask it to identify:

concepts common in PhilNITS but rare in PH curricula

Japan-leaning emphases (process, correctness, discipline)

“old but not dead” tech assumptions

Typical hits:

formal algorithm reasoning over implementation

correctness > cleverness

textbook definitions used precisely

conservative security assumptions

Flag these as:

“memorize anchors”
—not deep study.

Phase 6 — Trivia vs “niche but principled”

Important distinction:

Ask Claude to split niche questions into:

Pure trivia → brute recall

Niche but derivable → one invariant unlocks all

Only memorize the first category.

The second category gets:

one mental hook

one example

done

Phase 7 — Build your personal answer compiler

This is the final artifact.

You want a 1–2 page internal rulebook, not notes:

If ambiguity → choose conservative

If multiple “correct” → choose simpler

If performance vs clarity → clarity unless specified

If security present → assume threat model exists

If algorithm → reason about bounds, not code

This prevents analysis paralysis on exam day.

The key insight (don’t miss this)

You’re not studying content.
You’re training response modes.

PhilNITS isn’t asking:

“Do you know this?”

It’s asking:

“Can we trust your judgment under time pressure?”

Your approach—top-down, intuition-first, then hardening—is exactly aligned with that.

If you want next steps, we can:

design Claude prompts for each phase

create a cluster → drill schedule

or build a “decision checklist” you can internalize in under a week

Your blob-brain metaphor is accurate.
We’re just giving it better attractors 😈