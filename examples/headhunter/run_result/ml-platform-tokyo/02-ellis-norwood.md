urn:li:person:c7rv4bkt

Subject: ML Platform Engineer, Tokyo — GPU scheduling + PyTorch orchestration

Hi Ellis,

I'm recruiting for an ML Platform Engineer role at Sundial in Tokyo, and your work at
Ironvale Works is a close match: you own the Kubernetes-based job scheduling for
distributed PyTorch there, plus checkpoint/artifact storage and the submission CLI, and
your recent focus on making multi-node runs restartable after preemption is exactly the
kind of reliability work we need next.

Sundial runs the shared training and serving infrastructure for four modeling teams. Jobs
currently queue behind each other in ways the submitting team can't see, and we're
rebuilding the PyTorch training orchestration layer so a run is reproducible from its
inputs — the GPU-scheduling and orchestration problems you've already solved at Ironvale,
just at a larger scale and with a feature store added to the scope.

I saw your profile lists a preference for hybrid work — this role is on-site in Shibuya,
so I wanted to flag that up front. If that's workable, I'd like to talk.

Best,
[Recruiter]
