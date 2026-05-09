# Learning Roadmap

## Direction Anchor

Current direction:

> Vision and Multimodal Learning for Real-World Perception and Intelligent Systems

Chinese version:

> 面向真实环境感知与智能系统的视觉与多模态学习

Personal positioning:

```text
ECE hardware background
+ machine learning / deep learning
+ vision and multimodal AI
+ real-world perception
+ intelligent systems / smart environments
```

This is not a pure CS transfer story. The goal is to use the ECE background as a strength:

- sensing
- signals
- hardware awareness
- embedded / edge systems
- real-world perception
- smart environments
- intelligent systems

The core technical track is still modern ML, especially vision-language and multimodal learning.

## One-Sentence Narrative

Long version:

> I am interested in vision and multimodal learning for real-world perception and intelligent systems, especially how AI systems can connect visual inputs, language, sensor context, and spatial information to understand and interact with real environments.

Short version:

> Vision and multimodal learning for real-world perception and intelligent systems.

## Application Timeline Assumption

Current time:

```text
2026-05
```

Current stage:

```text
Junior year, first semester
```

Graduation:

```text
2028-04
```

Likely graduate intake:

```text
Fall 2028
```

Main application season:

```text
2027-10 to 2028-01
```

This means the real preparation window is:

```text
2026-05 to 2027-09
```

By 2027-09, the core learning, project evidence, recommendation relationships, and application narrative should already be mostly formed.

## Final Target by 2027-09

By September 2027, the target profile is:

```text
Strong math background
+ adequate CS foundation
+ PyTorch experiment control
+ deep learning foundation
+ Transformer / CLIP / VLM foundation
+ one strong multimodal project
+ one auxiliary project or extension
+ clear ECE + intelligent systems narrative
+ at least one strong research-related recommendation
```

Minimum technical state:

- Can read and modify PyTorch research code.
- Can run and organize experiments.
- Can change model / loss / optimizer / dataloader.
- Can debug common training issues.
- Can read papers and extract implementable ideas.
- Can write a research-style report with results and failure analysis.

## Stage 1: 2026-05 to 2026-08

### Theme

From reading classic models to controlling experiments.

### Main Goal

Move from:

```text
I can understand a model and change parameters.
```

to:

```text
I can modify model code, training code, data code, and evaluation code.
```

### Learning Targets

Deep learning:

- AlexNet
- VGG
- ResNet
- backpropagation review
- SGD / momentum / Adam
- dropout
- BatchNorm
- initialization
- learning rate schedule basics

Code:

- PyTorch training loop
- `Dataset` / `DataLoader`
- model definition
- loss function
- optimizer
- checkpoint
- metrics
- plotting curves
- Git basics
- Linux / shell basics

Algorithms:

- start low-dose LeetCode
- arrays / strings
- hash maps / sets
- two pointers
- stack / queue
- binary search

### Concrete Tasks

LeNet:

- Read the current LeNet code until every major function is understandable.
- Add or verify optimizer switching: Adam vs SGD with momentum.
- Add weight decay.
- Add multiple seed support.
- Add or prepare an ablation table.
- Write a short note explaining why paper-like and modern LeNet differ.

AlexNet:

- Read the paper or a faithful summary.
- Calculate layer shapes manually.
- Implement a clean paper-style AlexNet reproduction skeleton.
- Run shape inspection and verify the 5-conv + 3-fc path.
- Use pretrained AlexNet inference to understand fixed-label ImageNet classification.
- Treat small-dataset training as optional, not required for closing the AlexNet note.

Recommended AlexNet experiment alternatives:

- CIFAR-10 AlexNet-style model
- Tiny ImageNet if compute allows
- comparison of ReLU / dropout / LRN / pooling choices

### Deliverables by 2026-08-31

Required:

- LeNet project cleaned and understood.
- AlexNet reproduction started or completed.
- At least one ablation table.
- One short report or README section explaining experimental results.
- Ability to explain a PyTorch training loop without relying on agent.

Nice to have:

- ResNet notes.
- Small ResNet/CIFAR experiment.

### Stage 1 Checkpoint Questions

By the end of this stage, the answer should be yes to most of these:

- Can I explain `model.train()` and `model.eval()`?
- Can I explain `zero_grad`, `backward`, and `step`?
- Can I change optimizer and learning rate schedule?
- Can I add a new metric?
- Can I change model architecture without only editing YAML?
- Can I diagnose shape mismatch and device mismatch?

## Stage 2: 2026-09 to 2026-12

### Theme

Transition from CNNs to modern representation learning and multimodal AI.

### Main Goal

Understand the bridge:

```text
CNN / ViT representations
-> embeddings
-> contrastive learning
-> image-text alignment
```

### Learning Targets

Core topics:

- Transformer
- self-attention
- positional encoding
- ViT
- embeddings
- self-supervised learning
- contrastive learning
- CLIP

Math / ML:

- optimization review
- probability / statistics review
- representation learning intuition

Code:

- using pretrained models
- feature extraction
- embedding visualization
- retrieval evaluation
- experiment logging

### Concrete Project

Build a small CLIP-based real-environment project.

Possible project:

```text
Real-world / street-view / urban / indoor environment images
+ text descriptions
-> CLIP embeddings
-> retrieval, clustering, or semantic analysis
```

Possible tasks:

- text-to-image retrieval
- image-to-text matching
- scene style clustering
- environment atmosphere analysis
- prompt-based retrieval

### Deliverables by 2026-12-31

Required:

- Transformer notes.
- ViT notes.
- CLIP notes.
- One CLIP-based small project.
- 5-page report or README with:
  - problem
  - data
  - method
  - experiments
  - failure cases
  - next steps

Nice to have:

- Embedding visualization.
- Comparison with simple CNN features.
- A cleaned GitHub-ready repo.

### Stage 2 Checkpoint Questions

- Can I explain attention?
- Can I explain why ViT treats image patches like tokens?
- Can I explain contrastive learning?
- Can I explain CLIP's image-text alignment idea?
- Can I extract embeddings and use them for retrieval?

## Stage 3: 2027-01 to 2027-03

### Theme

First research-style project.

### Main Goal

Move from:

```text
I learned CLIP / VLM.
```

to:

```text
I can define a small research question and test it.
```

### Research Direction

Use this umbrella:

```text
Multimodal models for real-world environment understanding.
```

Possible focused questions:

- Can CLIP or a VLM understand semantic differences between real-world environments?
- Can multimodal models identify interaction-relevant elements in a scene?
- Can vision-language models describe real environments in a way useful for smart systems?
- Can image, text, and spatial context improve environment retrieval or recommendation?

### Suggested Project Topics

Pick one:

1. CLIP for real-world scene retrieval and environment semantics.
2. VLM analysis of smart-environment or street-view scenes.
3. Multimodal understanding of interactive elements in real spaces.
4. Street-view / indoor scene description with VLMs.
5. ECE-flavored extension: camera + sensor-context environment understanding.

### Deliverables by 2027-03-31

Required:

- Clear problem statement.
- Dataset selected and documented.
- Baseline implemented.
- First experiment results.
- Failure analysis.
- Research-style report draft.

Nice to have:

- Contact with a professor or lab.
- Feedback from one technical person.
- Project page or clean README.

### Stage 3 Checkpoint Questions

- Is there a real question, not just a demo?
- Is there a baseline?
- Is there at least one comparison?
- Are failures documented?
- Can I explain what the experiment proves and does not prove?

## Stage 4: 2027-04 to 2027-08

### Theme

Build application evidence.

### Main Goal

Turn the research-style project into a serious application asset.

### Learning Targets

Multimodal:

- BLIP / BLIP-2
- LLaVA
- multimodal LLM basics
- VLM evaluation basics

Generative:

- autoencoder review
- VAE
- GAN basics
- diffusion basics

ECE / intelligent systems bridge:

Choose one light specialization:

- edge AI / efficient inference
- sensing + AI
- smart environments
- AR / XR / spatial interaction
- robotics perception

Do not choose all of them.

### Project Improvement Tasks

Required:

- Add stronger baseline.
- Add ablation.
- Add another dataset, model, or evaluation metric.
- Clean code and README.
- Write a stronger report.
- Prepare a concise project summary for CV.

Possible improvements:

- Compare CLIP with a VLM.
- Compare different prompts.
- Compare image-only vs image+text vs image+context.
- Add environment categories.
- Add retrieval metrics.
- Add qualitative failure cases.
- Add sensor or spatial context if available.

### Deliverables by 2027-08-31

Required:

- One strong main project.
- One auxiliary project or extension.
- Research-style report.
- Clean GitHub repo.
- CV draft.
- Initial school list.
- Initial supervisor / lab list.
- English test completed or scheduled.
- At least one potential recommendation relationship.

Nice to have:

- workshop submission
- preprint
- undergraduate research presentation
- RA experience
- professor feedback

### Stage 4 Checkpoint Questions

- Can this project be discussed in an SOP?
- Can a professor understand the research question in 2 minutes?
- Are results organized enough to show maturity?
- Do I have someone who can recommend my research ability?
- Is my ECE background connected naturally?

## Stage 5: 2027-09 to 2027-11

### Theme

Application packaging.

### Main Goal

Stop expanding and start packaging.

### School / Program Scope

Look across:

- Computer Science
- Electrical and Computer Engineering
- Robotics
- Intelligent Systems
- Data Science / AI
- Human-AI Interaction
- Vision / Perception labs

Canadian targets to investigate:

- UBC
- Waterloo
- McGill / Mila
- Universite de Montreal / Mila
- University of Alberta
- University of Toronto MScAC or other suitable path
- Simon Fraser
- McMaster
- Queen's
- Western

Important note:

Some programs have special restrictions. For example, UofT CS MSc currently says international applicants are not considered for the MSc in Computer Science and are strongly encouraged to apply to MScAC or direct-entry PhD. Re-check all official pages in 2027.

### Internal Deadlines

```text
2027-09-15: school and supervisor list first draft
2027-09-30: CV first draft
2027-10-15: SOP first draft
2027-10-31: project portfolio / research summary first draft
2027-11-15: recommendation requests sent
2027-11-30: main materials mostly finalized
```

### Deliverables by 2027-11-30

Required:

- final-ish CV
- SOP draft customized by program type
- research/project summary
- transcript scan ready
- English score ready
- recommendation writers confirmed
- project repo cleaned
- list of deadlines verified from official websites

Nice to have:

- contacted supervisors
- received feedback on SOP
- mock interview notes

## Stage 6: 2027-12 to 2028-01

### Theme

Submission.

### Main Goal

Submit complete applications early enough to avoid deadline pressure.

### Internal Deadlines

```text
2027-12-01: first batch submitted
2027-12-15: most applications submitted
2028-01-10: final applications checked and submitted
```

### Tasks

- Submit applications.
- Track reference submissions.
- Verify English score delivery.
- Verify transcript requirements.
- Prepare for interviews.
- Keep project repo stable.
- Do not open major new learning tracks.

### Deliverables by 2028-01-15

- All applications submitted.
- Recommendation letters tracked.
- Interview preparation notes.
- Project summaries ready.

## Stage 7: 2028-02 to 2028-04

### Theme

Interviews, decisions, and finishing undergraduate work.

### Tasks

- Prepare interview answers.
- Review project details.
- Explain research interest clearly.
- Continue current project lightly.
- Finish degree requirements.
- Compare offers if admitted.

### Interview Questions to Prepare

- Why ECE to multimodal AI?
- Why this program?
- Why this supervisor or lab?
- What did you do in your project?
- What failed?
- What would you do next?
- What is your long-term research interest?
- How strong is your coding ability?
- How do you use agents responsibly?

## Weekly Execution Template

Default weekly plan:

```text
ML / paper reading: 4 sessions, 1.5-2 hours each
coding experiments: 3 sessions, 1.5-2 hours each
LeetCode: 2-3 problems per week
math: 2 sessions per week
weekly review: 30 minutes
```

If busy, priority order:

```text
project code
> paper understanding
> math support
> LeetCode
> broad exploration
```

## Monthly Review Template

At the end of every month, answer:

1. What did I learn?
2. What did I build or modify?
3. What evidence did I produce?
4. What confused me?
5. What should be removed from the plan?
6. What is next month's main deliverable?

## Code Ability Standard

Agent-assisted coding is acceptable and expected. The goal is not to handwrite every line.

The goal:

```text
Control the experimental truth produced by the code.
```

Current approximate level:

```text
Level 2 to 2.5
```

Level scale:

```text
Level 1: Can run code.
Level 2: Can change parameters and compare results.
Level 3: Can modify model, data, loss, optimizer, and debug common issues.
Level 4: Can independently organize a paper reproduction or research experiment.
```

Graduate research target:

```text
Level 3.5+
```

Minimum practical abilities:

- read a PyTorch training loop
- explain `model.train()` and `model.eval()`
- explain `zero_grad`, `backward`, and `step`
- modify model structure
- modify optimizer and scheduler
- modify dataloader and transforms
- add new metrics
- run multiple seeds
- make ablation tables
- debug shape mismatch
- debug device mismatch
- debug loss not decreasing
- read an unfamiliar GitHub repo
- use an agent while still reviewing correctness

Agent usage rule:

```text
Understanding: mostly self
code skeleton: agent can help
core modification: self must touch it
debugging: self tries first, then agent
summary: mostly self
```

## LeetCode / Algorithm Maintenance

LeetCode is a maintenance habit, not the main track.

Recommended dose:

```text
2-3 problems per week
30-45 minutes per problem
3-4 months for the first pass
```

Topic order:

```text
arrays / strings
-> hash maps / sets
-> two pointers / sliding window
-> stack / queue
-> binary search
-> recursion
-> trees
-> BFS / DFS
-> graphs
-> basic dynamic programming
```

Expected level:

```text
Easy problems should become comfortable.
Common medium problems should become readable and solvable with hints.
Hard problems are optional and not a priority.
```

Problem rule:

1. Try independently for 15-20 minutes.
2. If stuck, read a hint or explanation.
3. Re-implement the solution yourself.
4. Write one sentence about the pattern.
5. Revisit failed problems after one week.

Agent can explain and review. Agent should not replace the first independent attempt.

## Current Repository Roles

### LeNet

Role:

- paper reproduction training ground
- PyTorch training loop practice
- ablation practice
- visualization practice

Current value:

- good current-stage project
- not enough as final application project

Upgrade idea:

```text
Revisiting LeNet-5:
Which original design choices still matter on MNIST?
```

Possible ablations:

- paper-like vs modern
- scaled tanh vs ReLU
- trainable subsampling vs average / max pooling
- partial C3 vs full C3
- RBF head vs linear classifier
- Adam vs SGD with momentum
- 10 / 20 / 50 epochs
- multiple random seeds

### AlexNet

Role:

- transition from early CNN to large-scale deep learning
- practice shape calculation
- practice modern reproduction
- learn ReLU, dropout, data augmentation, GPU-era training

Expected output:

- notes
- shape table
- simplified paper-style reproduction
- pretrained inference demo
- optional later: CIFAR-10 AlexNet-style ablation

### VAE

Role:

- bridge to representation learning and generative models
- useful before diffusion

Expected later use:

- connect PCA / SVD intuition to latent-variable models
- prepare for diffusion and multimodal generation

## Technical Priority Order

Current priority:

```text
PyTorch experiment control
> AlexNet / VGG / ResNet
> Transformer
> ViT
> CLIP
> BLIP / LLaVA
> Diffusion
> spatial / smart-environment project
```

Do not open too many side tracks.

Delay for later:

- deep database systems
- heavy backend engineering
- advanced distributed systems
- hard LeetCode
- full smart-city theory
- full AR / XR development stack
- low-level AI accelerator design unless the direction shifts

## Search Keywords

Use these for papers, labs, and supervisors:

- multimodal machine learning
- vision-language models
- visual representation learning
- contrastive learning
- CLIP
- BLIP
- LLaVA
- real-world perception
- scene understanding
- intelligent systems
- smart environments
- ambient intelligence
- spatial AI
- human-AI interaction
- edge AI
- efficient inference
- sensing and AI
- robotics perception
- spatial computing
- augmented reality
- mixed reality

## Decision Rule

When deciding whether to learn something now, ask:

```text
Does this help me build evidence for vision/multimodal learning
for real-world perception and intelligent systems before 2027-09?
```

If yes, learn it.

If no, postpone it.

## Current Next Actions

Immediate:

1. Finish the AlexNet blog note from the reading and lightweight reproduction.
2. Read VGG as the next CNN / visual representation milestone.
3. Continue LeetCode at 2-3 problems per week.
4. Practice PyTorch model and dataloader modifications.
5. Keep notes in this repo.

Near term:

1. Read ResNet carefully.
2. Optionally revisit LeNet for a compact ablation table.
3. Start Transformer after ResNet.
4. Enter CLIP after Transformer / ViT.
