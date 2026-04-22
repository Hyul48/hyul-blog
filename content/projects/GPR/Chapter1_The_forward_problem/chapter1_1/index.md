---
title: "1.1 Notions of Electrodynamics in Material Media"
description: "Core electrodynamics concepts used in the GPR forward problem."
weight: 1
math: true
---

# 1.1 Notions of Electrodynamics in Material Media
1.1 절은 전자기 이론의 기본 개념과, 이후 수치적 구현에서 다루게 될 몇 가지 특수한 사항들을 소개한다. 특히 문헌에서 흔히 사용되지만 그 의미가 충분히 설명되지 않는 몇몇 가정들이 가지는 함의를 중점적으로 설명한다.

전자기학의 방정식과 그 물리적 해석은 우선 실제 물리 공간인 시간 영역에서 제시한다. 이후 "비물리적이지만 매우 유용한 푸리에 영역"인 주파수 영역으로 옮겨가 시뮬레이션과 역산을 수행한다.

또한 GPR 데이터를 역산하여 정량화하고자 하는 대상이 자연 매질의 전자기적 물성이므로, 이러한 매질의 전자기적 특성도 함께 논의한다. 이를 위해 해당 물성들을 매개변수화 할 수 있는 유전 응답 모델(dielectric response models)을 상세히 설명한다. 특히 지구 물리학에서 널리 사용되지만, 실수로 truncated form로 적용되면서 그 한계가 충분히 논의되지 않은 Jonshcer의 이른바 보편 응답(universal response)에 대해 언급한다. 아울러, 이후 역산 과정에서 사용할 보다 단순한 매개변수화 방식을 가정할 때 어떤 결과와 제약이 따르는지도 함께 논의한다.

## 1.1.1 Maxwells equation and constitutive relations
전자기파의 특성은 맥스웰 방정식(패러데이 법칙, 암페어 법칙, 가우스 법칙)에 의해 묘사될 수 있다.
$$
\nabla \times \mathbf{E}(\mathbf{r}, t)
=
- \frac{\partial \mathbf{B}(\mathbf{r}, t)}{\partial t},
\qquad \text{Maxwell-Faraday's equation}
\tag{1.1}
$$

$$
\nabla \times \mathbf{H}(\mathbf{r}, t)
=
\frac{\partial \mathbf{D}(\mathbf{r}, t)}{\partial t}
+
\mathbf{J}(\mathbf{r}, t),
\qquad \text{Maxwell-Amp\`ere's equation}
\tag{1.2}
$$

$$
\nabla \cdot \mathbf{D}(\mathbf{r}, t) = q(\mathbf{r}, t),
\qquad \text{Maxwell-gauss equation(electric)}
\tag{1.2}
$$
$$
\nabla \cdot \mathbf{B}(\mathbf{r}, t) = 0,
\qquad \text{Maxwell-gauss equation(magnetic)}
\tag{1.2}
$$

여기서 $\mathbf{E}$는 전기장으로, 단위는 $\mathrm{V/m}$이고, $\mathbf{H}$는 자기장으로 단위는 $\mathrm{A/m}$이다. 또한 $\mathbf{D}$는 전기 유도(electric induction) 또는 전기 변위(electric displacement)로서 단위는 $\mathrm{C/m^2}$이며, $\mathbf{B}$는 자기 유도로 단위는 $\mathrm{T}$이다.
$\mathbf{J}$는 전도 전류 밀도(conduction current density)로 단위는 $\mathrm{A/m^2}$이고, $q$는 전하 밀도(electric charge density)를 나타내며 단위는 $\mathrm{C/m^3}$이다. 변수 $\mathbf{r}$은 위치 벡터(position vector)로서 각 좌표의 단위는 $\mathrm{m}$이고, $t$는 시간으로 단위는 $\mathrm{s}$이다. 여기서 등장하는 모든 장과 변수들은 실수값(real quantities)이다.

소위 오른손 법칙(right-hand rule)에 따르면, 패러데이 법칙은 시간에 따라 변하는 자기 선속 $\mathbf{B}$가 $\mathbf{B}$를 중심으로 회전하는 전기장 $\mathbf{E}$를 생성함을 의미한다. 마찬가지로, 암페어-맥스웰 방정식은 전류 $\mathbf{J}$ 또는 시간에 따라 변하는 전기 선속 $\mathbf{D}$가 회전하는 자기장 $\mathbf{H}$를 생성함을 나타낸다. 한편, 맥스웰-가우스 방정식의 물리적 의미는 전하 밀도 $q$가 전기 전기 선속 $\mathbf{D}$의 원천이 된다는 것이며, 이에 대응되는 자기 선속의 원천은 존재하지 않는다는 점이다. 

**시간 영역 구성 관계(consitutive relations)**
