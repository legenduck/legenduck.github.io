---
title:          "HeartGuard: ECG-Based Heart Failure Prediction for Elderly Patients"
date:           2024-11-01 00:01:00 +0800
selected:       false
pub:            "2024 Yonsei University Health System Bigdata Challenge,"
pub_last:       ' <span class="badge badge-pill badge-publication badge-success">Excellence Award</span>'
pub_date:       "2024"
cover:          /assets/images/covers/ECG.png
summary: >-
  Heart failure risk prediction model using multimodal fusion of ECG waveforms and clinical data.
abstract: >-
  Developed a heart failure risk prediction model for elderly patients (65+) by effectively integrating heterogeneous ECG waveform data and Clinical Data Mart (CDM) records, focusing on QT interval prolongation as a key cardiac risk indicator. Designed a multimodal fusion architecture that processes ECG temporal patterns through a bi-directional 2-layer LSTM and CDM clinical features through XGBoost independently, then combines learned representations via a feed-forward neural network. This separate-then-fuse approach, rather than end-to-end neural architectures like TabNet, ensures balanced contribution from both modalities—preventing weight bias toward a single data source while respecting the medically established importance of both ECG signals and clinical records for heart failure prediction.
links:
---

