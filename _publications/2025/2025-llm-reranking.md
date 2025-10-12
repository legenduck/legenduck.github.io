---
title:          "Reasoning in Offline: Towards Efficient LLM Reranking for Real-Time Recommendation"
date:           2025-08-01 00:01:00 +0800
selected:       true
pub:            "Ongoing Research Project,"
pub_date:       "2025"
cover:          /assets/images/covers/Persona4Rec.png
summary: >-
  Developed a recommendation framework that performs offline LLM reasoning to construct interpretable persona representations of items, enabling real-time inference without expensive LLM calls. By transforming user-item relevance into user-persona matching through a lightweight encoder, achieved performance comparable to LLM-based rerankers while substantially reducing inference latency for practical deployment.
abstract: >-
  Recent advances in large language models (LLMs) offer new opportunities for recommender systems by capturing the nuanced semantics of user interests and item characteristics through rich semantic understanding and contextual reasoning. In particular, LLMs have been employed as rerankers that reorder candidate items based on inferred user–item relevance. However, these approaches often require expensive inference-time reasoning, leading to high latency that hampers real-world deployment. In this work, we introduce Persona4Rec, a recommendation framework that performs offline reasoning to construct interpretable persona representations of items, enabling lightweight and scalable real-time inference. In the offline stage, Persona4Rec leverages LLMs to reason over item reviews, inferring diverse user motivations that explain why different types of users may engage with an item; these inferred motivations are materialized as persona representations, providing multiple, human-interpretable views of each item. Unlike conventional approaches that rely on a single item representation, Persona4Rec learns to align user profiles with the most plausible item-side persona through a dedicated encoder, effectively transforming user–item relevance into user–persona relevance. At inference time, this persona-profiled item index allows fast relevance computation without invoking expensive LLM reasoning. Extensive experiments show that Persona4Rec achieves performance comparable to recent LLM-based rerankers while substantially reducing inference time. Moreover, qualitative analysis confirms that persona representations not only drive efficient scoring but also provide intuitive, user-grounded explanations. These results demonstrate that Persona4Rec offers a practical and interpretable solution for next-generation recommender systems.
links:
---

