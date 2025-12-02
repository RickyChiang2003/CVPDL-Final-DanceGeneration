# Related Paper Review
負責人：江融其

## Background
本專案想完成的任務是 In-between Dance Generation: 輸入一定長度的頭尾兩段 3D 舞蹈動作骨架，生成中間一定長度的舞蹈動作，且需在生成高品質舞蹈動作的同時自然銜接前後動作。   

靈感來自於舞蹈的編舞。在編舞時，通常首要思考的便是舞蹈動作是否契合音樂、節奏和舞風，接著便是確認前後動作的連貫性。實際編舞者在將自身熟悉的舞蹈動作風格 ── 也就是舞風 ── 帶入音樂中時常會遇到的問題是動作與動作間銜接不順暢，或是在想到前後幾段的動作後難以在它們之間加入能連接兩者的中間動作。這類問題會在一支舞蹈有多名編舞者時更加明顯，因為即便雙方擅長的舞風類別相同，也常面臨動作偏好不一致而難以共編的情況。   

這種生成中間資料片段的任務被稱為 In-between Generation 。概念出自於動畫繪師常做的 Tweening ，意思是在兩關鍵畫面 (Keyframes) 中間加入銜接用的動漫畫面 (In-between) 。 In-between Generation 旨在透過輸入頭尾資料 (這裡就是頭尾各輸入一支動作骨架) 生成能串連頭尾的中間空缺的資料流。    

在 Motion Generation 中， In-between 相關的任務會給定動作的 prefix 與 suffix frame 並生成中間缺失的動作片段。除了要平滑連接，還需符合人體運動的物理學。 Dance Motion 更比單純 Motion 複雜得多，除了要有相應的舞蹈骨架標註資料集，考慮運動的合理性與物理學，還要考慮其對應的音樂旋律、配合節奏、契合所需的舞蹈風格等 (像是芭蕾與嘻哈的舞蹈風格顯然差異甚大) 。近年來已經能透過文字、音樂與圖片等多模態輸入生成穩定的動作骨架，取得了區分舞風類別的能力，也已設計出如 Kinetic FID 或 beat alignment score 等算分標準來評價舞蹈生成的品質。   

然而，考慮到使用者需求，輸入可能不只需要既有的文字與圖片，輸入有一定時長的頭尾舞蹈動作骨架片段來生成中間片段的銜接動作，對於編舞者而言將更加實用，但也更加考驗目前的模型，因為目前尚未有同時應用了 Music 與 Motion 資訊的 In-between Dance Generation 研究。當前的編舞者們若想生成銜接用的舞蹈動作，仍只能將需求以文字或圖片輸入給多模態模型，然後再期待模型能生成有用的動作建議 ── 當然，在沒有輸入頭尾動作的情況下往往效果不彰。   

根據我們的瀏覽與近年的相關 Survey Research， Motion Generation 這種被廣為研究的任務，其架構創新和資料集數量都遠非進階應用型任務 Dance Generation 所能相比。因此，雖然 Motion Generation 相關模型缺少關於音樂與節奏的考量，但其動作骨架的分析很完善，使得 Music-to-Motion 或是 Text-to-Motion 的生成任務經常參考 Motion Generation 的架構和方法，並且實驗也證實許多概念套用於 Dance Generation 上後都能取得不錯的成效。    

因此，我們提出了第一個完整的 In-between Dance Generation 架構。分別基於 Transformer 與 DiT (Diffusion Transformer) 兩種架構方向進行實驗，同時嘗試加入 RAG (Retrieval-Augmented Generation) 來強化 few shot training 與 evaluation matrix 方法，生成高品質的 in-between 舞蹈動作骨架。   

## Transformer


## DiT (Diffusion Transformer)



## Paper Lists
### Latest Motion / Human Motion Generation Survey
| Paper Name | Release Time | Link |
|:--|:--:|:--:|
| Human Motion Generation: A Survey | 2023/07 | [Link](https://arxiv.org/abs/2307.10894) |
| Motion Generation: A Survey of Generative Approaches and Benchmarks | 2025/07 | [Link](https://arxiv.org/abs/2507.05419) |
| A Survey of Human Motion Video Generation | 2025/09 | [Link](https://arxiv.org/abs/2509.03883) |

### Dataset



### Related Works


