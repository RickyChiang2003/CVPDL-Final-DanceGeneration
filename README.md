<br/>
<p align="center">
  <h1 align="center">CVPDL Final Project: In-between Dance Generation</h1>
  <p align="center">
    <a href="">吳克洋</a>,
    <a href="https://github.com/JunTingLin">林俊霆</a>,
    <a href="">陳郁仁</a>,
    <a href="">吳政霖</a>,
    <a href="">江融其</a>
  </p>
</p>


# Introduction
本專案為台大 2025 電腦視覺實務與深度學習課程的期末專題研究。   
<details>
<summary><b>點擊展開</b></summary>
    
細節請見 [`SURVEY.md`](https://github.com/RickyChiang2003/CVPDL-Final-DanceGeneration/blob/main/SURVEY.md) 與各 [`Research`](https://github.com/RickyChiang2003/CVPDL-Final-DanceGeneration/tree/main/Research) 內容   

## Summary
現有的 Dance Generation (舞蹈動作生成) 與 Motion Generation (生活類動作生成) 研究已經能透過文字、音樂與圖片等多模態輸入生成穩定的 3D 人體動作骨架，除了能區分舞蹈的風格 (舞風，如 Hip Hop, Ballet 等)，也已設計出如 Kinetic FID 或 beat alignment score 等評判標準來評價舞蹈生成的品質，研究方法遍及 transformer based, diffusion based 等的重要的生成式模型架構。   

而在發展歷時已久的 Motion Generation 中有個任務叫做 In-between Generation ，旨在透過輸入頭尾資料流 (這裡就是頭尾各輸入一支舞蹈動作骨架"影片") 生成能串連頭尾的中間空缺的資料流。概念出自於動畫繪師常做的 Tweening ，意思是在兩關鍵畫面 (Keyframes) 中間加入銜接補充用的畫面 (In-between) 。近幾年 In-between Generation 在 Motion 、 Music 相關的領域中均有所突破，但作為結合了音樂與動作資訊的應用領域的 In-between Dance Generation 卻尚未有相關研究。   

我們提出了第一個完整的 In-between Dance Generation 架構。分別基於 Transformer 與 DiT (Diffusion Transformer) 兩種架構方向進行實驗：   

- Transformer 架構基於 ......    

- DiT 架構基於 [MDM: Human Motion Diffusion Model](https://guytevet.github.io/mdm-page/)     

另外，我們也構想以 RAG (Retrieval-Augmented Generation) 來強化 few shot training 與 evaluation matrix 方法，生成高品質的 in-between 舞蹈動作骨架。

</details>



/>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   


以下為各方法的實作範例：   

|  | `BAMM-Unimumo` | `MDM` | `` |
|:--:|:--:|:--:|:--:|
|  |  |  |  |
|  |  |  |  |
| score |  |  |  |
| FID |  |  |  |

/<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   

# File Structure 
以下為本專案架構，有成功的方法會被置於 `Research/` 資料夾中：  

```bash
CVPDL-Final-DanceGeneration/
│   
├── README.md
├── SURVEY.md            # related paper review
├── Research/            # successful research
│   ├── BAMM-Unimumo/
│   ├── MDM/
│   └── RAG/
└── Research-abandon/    # abandoned research
...
```

# Usage
各 `Research` 方法皆附有自己的 `READEME.md` ，請移至各自的路徑。

# Work Assignment
- 吳克洋：破台、  100%
- 林俊霆：破台、  100%
- 陳郁仁：破台、  100%
- 吳政霖：破台、  100%
- 江融其：破房、 -300%
