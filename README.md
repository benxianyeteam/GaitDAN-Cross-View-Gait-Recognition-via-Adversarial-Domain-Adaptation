<img width="1337" height="848" alt="image" src="https://github.com/user-attachments/assets/321daeae-a4d0-40f0-bd27-0040818362a6" />                                                       
															
															Abstract

  The task of cross-view gait recognition is challenged by significant appearance differences caused by varying camera viewpoints. Existing methods address this by either transforming gait features to a common view or eliminating view-specific information, but these approaches often struggle with unknown camera angles. To overcome these limitations, this paper proposes GaitDAN, a novel approach that treats cross-view gait recognition as a domain adaptation problem. We introduce an Adversarial View-change Elimination (AVE) module and a Hierarchical Feature Aggregation (HFA) strategy to effectively eliminate view-dependent features while maintaining discriminative gait representations. Our approach uses adversarial domain adaptation to align gait features across multiple sub-domains (camera views) and achieve view-invariant feature representations. Extensive experiments on the CASIA-B, OULP, and OUMVLP datasets demonstrate the superiority of GaitDAN, achieving state-of-the-art performance in cross-view gait recognition under various walking conditions.

									                    	  摘要
  跨视角步态识别面临着由于相机视角变化引起的外观差异问题。现有的方法通常通过将步态特征转换为统一视角或消除视角特有的信息来应对这一挑战，但这些方法在面对未知相机角度时往往表现不佳。为了克服这些局限性，本文提出了一种新的方法——GaitDAN，将跨视角步态识别视为一个领域适配问题。我们提出了对抗性视角变化消除（AVE）模块和分层特征聚合（HFA）策略，有效地消除了与视角相关的特征，同时保持了判别性步态表示。我们的方法利用对抗领域适配对齐多个子领域（相机视角）中的步态特征，从而实现视角不变的特征表示。通过在CASIA-B、OULP和OUMVLP数据集上的大量实验，验证了GaitDAN的优越性，在各种步态条件下的跨视角步态识别任务中达到了最先进的性能。



