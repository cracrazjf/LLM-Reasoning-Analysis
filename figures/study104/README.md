# The 104-question study: figures and tables

Qwen3-8B and Qwen3-32B on 104 MedXpertQA two-option diagnosis questions (data/tasks/study104.csv: one question per source
question, 52 shown with the correct option as A and 52 as B), 60 thinking traces per question and model. Every figure is made by
`src/study104_figures.py <command>` and has its numbers next to it as CSV.

## fig1_accuracy.png (`accuracy`)

Accuracy of each model averaged over the 104 questions, without thinking (one greedy answer per question with thinking switched
off: the share of questions answered correctly) and with thinking (the mean over questions of the share of the 60 traces that
answer correctly); whiskers = 95% intervals from resampling questions; dashed line = 0.5. Qwen3-8B: 0.55 without thinking, 0.70
with thinking; Qwen3-32B: 0.67 and 0.77. All 60 traces right on 46 (8B) and 47 (32B) questions, at most half right on 31 and 23,
none right on 8 and 7. Per question, the two models' accuracies correlate (Spearman 0.72); the 32B is higher on 41 questions,
lower on 22 and equal on 41. One 8B trace reached the 32,768-token limit without an answer and is left out (6,239 answers; 32B
6,240). Numbers: fig1_accuracy.csv (per question), fig1_accuracy_summary.csv (per model and condition).

## fig2_answer_change.png (`change`)

How many questions reasoning changes the answer of, for (a) Qwen3-8B and (b) Qwen3-32B. Before reasoning = the first reaction
(one greedy answer with thinking switched off); after reasoning = the answer of the majority of the question's 60 thinking
traces (no question splits 30-30). Changed: the two answers differ, so the question goes from wrong to right or from right to
wrong; unchanged: they agree, right both times or wrong both times. Grey = right after reasoning, white = wrong after reasoning;
numbers = questions. Qwen3-8B: 24 changed (20 wrong to right, 4 right to wrong), 80 unchanged (53 right, 27 wrong). Qwen3-32B:
19 changed (15 wrong to right, 4 right to wrong), 85 unchanged (66 right, 19 wrong). For 8 (8B) and 10 (32B) questions the majority
is thin (between 18 and 42 of the 60 traces right). Numbers: fig2_answer_change.csv.

## fig3_conclusion_step.png (`steps`)

The change that the conclusion sentence makes to the closed readout, against the changes made by the sentences before it.
Confidence = log p(final letter) - log p(other letter) of the closed readout (the thinking cut at a sentence start and ended the
way the model ends it); the change across a sentence is the readout after it minus the readout before it, the first sentence
counted from the empty thinking block; the conclusion's change is the readout at the answer position minus the readout at the
onset of the conclusion sentence. 6,239 (8B) and 6,240 (32B) traces. (a) Median change (marker) and quartiles (line) at each of
the last 15 sentences before the conclusion and at the conclusion, traces aligned on their conclusion sentence. Before the
conclusion the median change of a sentence is 0 to 0.25 nats (8B) and 0.12 to 0.5 (32B), 45-62% of the changes are positive,
and the quartiles lie between -0.6 and +1.6; across the conclusion the median is 2.25 (quartiles 1.5 to 3.5) and 2.38 (1.0 to 4.1),
positive in 96% and 89% of traces. (b) For every trace, the share of its own earlier sentences whose absolute change is smaller
than the conclusion's change. Median 0.94 (8B) and 0.90 (32B); at least 0.9 in 62% and 51% of traces; the conclusion's change is
the largest of the whole trace in 20% and 27%. The bar at 0 holds the traces whose conclusion did not raise the readout (4% and
12%). Over all earlier sentences the median absolute change is 0.38 (8B) and 0.62 (32B). Numbers: fig3_conclusion_step.csv
(profile), fig3_conclusion_step_summary.csv.

## fig4_examples.png (`examples`)

The closed readout along three traces of one question in each model: Text-699 shown with the correct option first (A left
unilateral sacral extension, B left unilateral sacral flexion); left Qwen3-8B, right Qwen3-32B; rows the short, medium and long
trace. y = log p(A) - log p(B), the confidence in the final answer (all six traces answer A); x = thinking tokens kept before the
cut; small dots = the readout at every sentence start, the first at the empty thinking block; open circle = at the onset of the
conclusion sentence; filled circle = after it, at the model's own stop; the bracket gives the change across the conclusion
sentence; the quote is the model's own conclusion sentence. The change falls with length in both models: Qwen3-8B +19.5 (405
tokens, 23 sentences), +10.2 (862 tokens, 49 sentences), +3.0 (2,092 tokens, 135 sentences); Qwen3-32B +11.1 (436, 25), +4.6
(657, 33), +2.1 (1,804, 108). Before the conclusion the readout is 0.75 to 6.75 nats. Choice: chosen for illustration, on the
first 30 traces of each question. Of the 35 questions in which both models have a short, a medium and a long correct
letter-verdict trace (one from each third of the question's lengths) whose change falls with length for the own conclusion and
for both written sentences (fig5), this question has the largest changes and the lowest readouts before the conclusion; within
it the three traces are spaced about evenly on a log scale of length. Its within-question correlation between length and the
change (all 60 traces: Spearman -0.45 in the 8B, -0.33 in the 32B) is stronger than the typical one (median over the 104
questions -0.16 and -0.17, fig6). It is one of the 12 questions of the 12-question study, and its example question. Numbers:
fig4_examples.csv.

## fig5_examples_written.png (`examples_written`)

The same six traces up to the onset of their conclusion sentence, ended instead by a written sentence: "Therefore, the answer is
A." (black circle) and "Therefore, between A and B, the answer is A." (grey square), each followed by the model's own close; the
numbers give the change from the onset (open circle). The written sentences are 7 and 12 tokens long; their endpoints are drawn
apart along x only so that they can be told apart. The change still falls with length in both models and for both sentences:
"Therefore" +16.8, +10.2, +2.8 (8B) and +8.8, +4.9, +2.1 (32B); "between" +17.2, +9.5, +2.8 (8B) and +9.9, +4.6, +2.4 (32B). The
8B's medium trace and the 32B's long trace already end with "Therefore, the answer is A.", so their own and written changes
coincide. Numbers: fig5_examples_written.csv.

## fig6_per_question_correlation.png (`perquestion`)

Distribution over the 104 questions of the within-question Spearman correlation between a trace's thinking length and the change
across its conclusion sentence (own conclusion; all answered traces of the question, 59 or 60), (a) Qwen3-8B and (b) Qwen3-32B.
Bars: the percentage of the questions whose correlation falls in each bin; vertical line: zero. Qwen3-8B: median -0.16, negative in 75 of 104 questions; Qwen3-32B: median
-0.17, negative in 85. Per question (from resampling its traces, 1,000 draws), the 95% interval lies below 0 in 31 (8B) and 33
(32B) questions and above 0 in 2 and 1, against about 2.6 each by chance (from shuffling the lengths among a question's traces, 1,000 shuffles per question). The correlations vary across questions more than
sampling alone makes them vary (SD across questions 0.22 against a within-question sampling SD of 0.13 in the 8B, 0.17 against
0.13 in the 32B), so the questions differ in the size of the effect, more in the 8B than in the 32B. Numbers:
fig6_per_question_correlation.csv (per question: correlation, interval, shuffled range), fig6_per_question_correlation_summary.csv.

## fig7_attention_answer.png (`attention`)

Where the model looks when it writes the answer, with thinking. For each model, the upper bar splits the context into four parts by
their share of its tokens, the lower bar by their share of the attention of the token that writes the answer letter (the line break
after the model's own </think>; src/attention_at_answer.py run104 / analyze104: the attention weights of that token recomputed in
every layer from its query and the keys, averaged over layers and heads). Parts: the question (case text, both options and the
instruction), the reasoning before the conclusion sentence, the final conclusion sentence, and </think> with the line break and the
chat-template tokens. The first token of the context takes 45% (Qwen3-8B) and 60% (Qwen3-32B) of the attention and writes little
into the output (19% and 28% once every weight is multiplied by the size of what its token writes); it is left out of both bars and
the rest renormalised. Shares are averaged over a question's 60 traces, then over the 104 questions. The final conclusion holds
0.9% (8B) and 1.4% (32B) of the tokens and receives 7.4% and 8.8% of the attention, 8 and 6 times its share of the tokens; the
reasoning before it holds 80% and 75% of the tokens and receives 36% and 44%. The question receives about its share of the tokens
(22% of the attention for 18% of the tokens in the 8B, 24% for 22% in the 32B); without thinking it receives 47% and 60% of the
attention. </think>, the line break and the template tokens, about 1% of the tokens, receive 35% and 23%. Numbers:
fig7_attention_answer.csv (per model, condition and question: the share of the tokens and of the attention of each part).

## fig8_length_attention_jump.png (`mechanism`)

Longer thinking gives the final conclusion a smaller share of the answer's attention, and the conclusion raises the confidence in
proportion to that share. Within every question its 60 traces are split into fifths, (a) of thinking length and (b) of the share of
the answer-writing position's attention that falls on the conclusion sentence (as in fig7: the first token left out, mean over
layers and heads). Each point is the median over the 104 questions of the question's median in that fifth; whiskers = quartiles
over questions. Black = Qwen3-8B, grey = Qwen3-32B. (a) The conclusion's share falls from 8.2% in the shortest fifth to 6.5% in the
longest (8B) and from 10.1% to 7.7% (32B); within-question Spearman between length and share, median -0.38 and -0.40, negative in
101 of 104 questions in both models. (b) The change across the conclusion (the closed readout toward the final answer with the
conclusion minus without it, as in fig6) rises from 1.5 nats in the fifth with the smallest share to 3.8 in the largest (8B) and
from 0.9 to 4.4 (32B); within-question Spearman between share and change, median +0.60 in both, positive in 103 and 104 of 104
questions. The share accounts for the length effect: regressing the change on log2 thinking length within question (both demeaned
within question; 95% intervals from resampling questions) gives -0.69 nats per doubling [-0.91, -0.49] in the 8B and -0.63
[-0.79, -0.48] in the 32B; with the share added, the length term becomes +0.08 [-0.08, 0.24] and +0.58 [0.44, 0.71], while the share
contributes +0.72 [0.61, 0.85] and +0.89 [0.82, 0.97] nats per percentage point. Numbers: fig8_length_attention_jump.csv (the points),
fig8_length_attention_jump_stats.csv (correlations and regressions).

## fig9_masking_conclusion.png (`masking`)

Hiding the conclusion sentence from the answer gives the confidence of a trace whose conclusion was never written; hiding random
earlier sentences leaves the confidence unchanged. Confidence = log p(final letter) - log p(other letter) at the answer-writing
position, the final letter being the trace's own answer. Full reasoning, random earlier sentences hidden and conclusion sentence
hidden come from one masking pass per trace (src/ablate_conclusion.py run104 / analyze104): the hidden tokens are removed from the
attention of </think> and of the answer-writing position in every layer, everything before </think> left as it is. The random
earlier sentences are drawn from the sentences before the conclusion that do not name an answer, in a seeded order, until they
hold at least as many tokens as the conclusion (median 20 tokens against the conclusion's 8 in Qwen3-8B, 22 against 11 in
Qwen3-32B). Reasoning ended before the conclusion sentence = the thinking before that sentence, closed the way the model closes its
thinking, read without masking (the conclusion readout). Points = median over the 104 questions of each question's median over its
60 traces; lines = quartiles over questions. Qwen3-8B: full 11.7 nats, random earlier sentences hidden 11.7, conclusion hidden 8.1,
ended before the conclusion 8.3. Qwen3-32B: 13.8, 13.8, 10.2, 10.2. Per question, the confidence with the conclusion hidden lies
0.75 nats (8B) and 0.38 nats (32B) below the confidence with the conclusion never written (medians; within 1 nat in 75 and 79 of
the 104 questions). Numbers: fig9_masking_conclusion.csv (per question), fig9_masking_conclusion_summary.csv.

## fig10_conclusion_accuracy.png (`accuracy_conclusion`)

Hiding the conclusion sentence from the answer leaves the accuracy almost unchanged. Accuracy = the share of a question's 60 traces
whose answer is correct, averaged over the 104 questions; whiskers = 95% intervals from resampling questions; dashed line = 0.5.
With the conclusion sentence the answer is the trace's own; with it hidden (as in fig9) the answer is the letter the masked readout
prefers. Qwen3-8B: 0.70 with, 0.69 hidden (difference -0.016, 95% interval -0.036 to -0.000); Qwen3-32B: 0.77 and 0.77 (difference
-0.000, -0.004 to 0.003). Hiding the conclusion changes 205 of 6,239 answers in the 8B (152 from right to wrong, 53 from wrong to
right) and 53 of 6,240 in the 32B (27 and 26). Numbers: fig10_conclusion_accuracy.csv (per question),
fig10_conclusion_accuracy_summary.csv.

## fig11_conclusion_confidence.png (`confidence_conclusion`)

Hiding the conclusion sentence from the answer lowers the confidence in the final answer by about 3 nats. Confidence = log p(final
letter) - log p(other letter) at the answer-writing position, the final letter being the trace's own answer; bars = median over the
104 questions of each question's median over its 60 traces; whiskers = quartiles over questions. Qwen3-8B: 11.7 nats with the
conclusion sentence, 8.1 with it hidden; Qwen3-32B: 13.8 and 10.2. Per question the confidence falls by 3.3 nats (8B; quartiles 2.5
to 4.1) and 3.2 nats (32B; 2.2 to 4.3), and it falls in 103 of the 104 questions in both models. Numbers:
fig11_conclusion_confidence.csv (per question), fig11_conclusion_confidence_summary.csv.
