---
slug: introducing-ailoy
title: Introducing Ailoy
authors: jhlee
tags: [ailoy]
---

# Introducing Ailoy: A Drop-in Library for LLM & Agent Development

We’re excited to announce the launch of **Ailoy**, a drop-in library for LLM & agent development.

Large Language Models (LLMs) and AI agents have recently emerged as a major trend in the software industry, drawing significant attention.
Yet for many developers, even building a proof of concept can be daunting.
It often requires deep knowledge of machine learning, experience with parallel programming.
Cloud APIs can ease the burden, but they introduce new cost concerns.
Taking it to a production level? That’s even harder.

<!-- truncate -->

**Ailoy** is designed to solve these challenges.
We focused on on-device AI, which avoids complex setups and cloud costs by running entirely on the local machine.
But to access high-end AI capabilities, cloud APIs still play an important role.
Our goal is to keep the right balance between the two.
**Ailoy** works seamlessly across cloud, on-device, and even hybrid environments, helping you build and deploy advanced AI functionality with minimal overhead.

You don’t need heavy infrastructure. You don’t need a degree in AI. Just the right tools to get things done.

## 🚀 What is Ailoy?

Ailoy is a lightweight library that makes it easy to embed AI into your software.
Whether you're building a chatbot, an intelligent agent, or enhancing an existing system with LLM capabilities, Ailoy gives you the tools to do it with minimal effort.

It provides:

- High-level APIs to run LLMs, including multi-turn conversations, system prompt customization, and reasoning
- Built-in tools and functions for constructing agent-based systems, including *Model Context Protocol (MCP)*
- Easy agent creation and simplified integration of *Retrieval-Augmented Generation (RAG)*

Think of it as your AI development toolkit.
It's ideal for embedded assistants, workflow automation, or even building your own reasoning system from the ground up.

## 💡 Why Ailoy?

**1. Easy to Use**

No need to wrestle with complex orchestration frameworks or deep ML stack knowledge. With Ailoy, you can just import and go.
Ailoy is designed to feel like any modern developer library.

**2. Cloud or On-Device**

Ailoy supports both cloud vendors like OpenAI and on-device inference using optimized runtimes via TVM.
Even better, you can mix both seamlessly in a single application.

Want to run models offline to save on API costs? You can.
Need to fall back to the cloud when local resources aren't enough? That works too.
Or maybe you're building a hybrid that combines local tools with remote reasoning — Ailoy makes that easy.
It's also a great fit for secure or privacy-sensitive environments where cloud access isn't an option.

**3. Versatility Across Architectures**

Ailoy supports Windows, macOS, and Linux.
And it works with Python, Node.js and C++ environment.

We're continuously expanding our platform and language coverage to make Ailoy available wherever developers build.

**4. Developer-Centric Design**

We built Ailoy not only for ML engineers, but also all developers.
Many existing LLM frameworks require expertise in ML systems or infrastructure.
Ailoy aims to break that barrier.
Whether you’re writing a script, building a product, or shipping a SaaS, Ailoy lets you focus on your application logic.

We’re currently in the early stages, but iterating rapidly to support the features and workflows real developers care about.

## 🔍 What You Can Build with Ailoy

- A multi-turn chatbot powered by a local model and cloud
- A command-line assistant with custom tools
- A personalized agent with access to private files via RAG
- A desktop app that ships with an embedded LLM

## ⚠️ Current Limitations

While Ailoy strives for universality, there are a few constraints to keep in mind.

LLMs are still resource-intensive.
On-device execution requires a capable system, and performance can vary significantly across platforms.

On `macOS`, Apple silicon has made it relatively easy to run LLMs locally thanks to it's unified memory architecture and `Metal`.
However, older MacBooks may struggle to meet the performance demands.
On `Windows` and `Linux`, we currently rely on `Vulkan` for LLM execution, where GPU acceleration is often essential for achieving usable inference speeds.
As for mobile devices, we believe it's still too early for practical on-device LLM execution due to hardware limitations.
Inference speed and memory usage depend heavily on the model architecture and the capabilities of the target device.

That said, we’re optimistic about the future.
Many hardware vendors are now shipping computers with AI accelerators, and the landscape is quickly evolving.
We’re committed to making Ailoy compatible with these emerging platforms and enabling smooth on-device AI experiences as the hardware ecosystem matures.

## 🌐 What’s Next

We have a roadmap to make Ailoy even more accessible and powerful.

Currently, we support Alibaba’s `Qwen3` and OpenAI’s `GPT-4o`, and we plan to expand support to a broader range of models, including multimodal ones. 
We're also working to enable speech recognition and text-to-speech (TTS) models, opening up more possibilities for voice-based AI interactions.
In addition, we're bringing Ailoy to the web by supporting client-side execution in modern browsers.
Language support is also expanding beyond Python and JavaScript, with Java, C#, and other ecosystems on the horizon.
We’re also planning to support inter-machine scenarios.
By building remote model execution capabilities over TCP, Ailoy will enable AI agents to communicate and collaborate across machines.
This will make it possible to build distributed agent systems that can leverage multiple machines working together.

Our long-term goal is to make local-first AI the default — where on-device and cloud-based models work together seamlessly.
This hybrid approach empowers developers to build advanced AI applications while minimizing costs, protecting user privacy, and optimizing performance.

## 🤝 Join Us on the Journey

We believe accessible AI infrastructure can unlock a whole new class of applications.
Whether you're an indie developer, a startup team, or part of a large organization, Ailoy is here to help you:

- Streamline your development process with intuitive APIs and minimal setup
- Build and ship on-device AI applications without worrying about high cloud costs
- Run AI in environments where privacy, offline access, or security make cloud services impractical

As we continue expanding Ailoy’s capabilities, we invite you to be part of the journey.
Your feedback, ideas, and use cases will shape what Ailoy becomes.

**Ready to build?**

Grab the package, explore the docs, and start bringing your ideas.
