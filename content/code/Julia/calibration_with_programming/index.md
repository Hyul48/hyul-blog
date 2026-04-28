---
title: "Julia Calibration"
description: "Julia Calibration with SOLT Algorithm"
date: 2026-04-28
weight: 1
math: true
---

>해당 포스트는 Calibration에 대한 이해를 필요로 합니다. Calibration에 대한 이해가 없으신 분들은 [해당 포스트](../../../notes/VNA/calibration_2/)를 먼저 읽어주시길 바랍니다.

# Calibration
![이미지1](both.png)
이번 프로젝트에서 관심있는 파라미터는 $S_{11}$ only이므로 One-Port Calibration을 수행하는 코드를 짜보도록 하자. 먼저 1-Port Cal은 아래의 에러텀들을 측정해주어야 한다. 1-Port Cal에서 우리가 측정해 주어야 하는 에러텀들은 $e_{00}, e_{11}, e_{10}e_{01}$임을 확인했었다.
위의 에러텀들을 구할 수 있는 3가지 연립 방정식은 3개의 기준으로 엄밀하게 방정식을 세우면 다음과 같다.


- Open (개방)커패시턴스: $C_e(f) = C_0 + C_1f + C_2f^2 + C_3f^3$ 
    - 임피던스: $Z_{Open}(f) = \frac{1}{j 2\pi f C_e(f)}$ (이후 오프셋 지연 반영) 
    - 반사 계수: $\Gamma_{Open}(f) = \frac{Z_{Open}(f) - Z_0}{Z_{Open}(f) + Z_0}$
    $\quad$
- Short (단락)인덕턴스: $L_e(f) = L_0 + L_1f + L_2f^2 + L_3f^3$  
    - 임피던스: $Z_{Short}(f) = j 2\pi f L_e(f)$ (이후 오프셋 지연 반영) 
    - 반사 계수: $\Gamma_{Short}(f) = \frac{Z_{Short}(f) - Z_0}{Z_{Short}(f) + Z_0}$ 
    $\quad$
- Load (부하)
    - 임피던스: $Z_{Load}(f)$ = $R + j(2\pi f L_{series} - \frac{1}{2\pi f C_{shunt}})$ (이상적인 경우 $Z_0$(50Ω)에 가깝습니다.)
    - 반사 계수: $\Gamma_{Load}(f) = \frac{Z_{Load}(f) - Z_0}{Z_{Load}(f) + Z_0}$
    $\quad$

$$ 
\Gamma_{IN, Open} = e_{00} + \dfrac{e_{10}e_{01}\mathbf{\Gamma_{Open}(f)}}{1 - e_{11}\mathbf{\Gamma_{Open}(f)}}$$

$$\Gamma_{IN, Short} = e_{00} + \dfrac{e_{10}e_{01}\mathbf{\Gamma_{Short}(f)}}{1 - e_{11}\mathbf{\Gamma_{Short}(f)}}$$

$$\Gamma_{IN, Load} = e_{00} + \dfrac{e_{10}e_{01}\mathbf{\Gamma_{Load}(f)}}{1 - e_{11}\mathbf{\Gamma_{Load}(f)}} $$

여기서 우리가 알아야할 에러 항목들($e_{00}, e_{11}, e_{10}e_{01}$) 주파수에 의존적이며 복소수임을 알 수 있다.($\because \Gamma$가 주파수에 의존적이며 복소수이기 때문 )

먼저 OnePortCal 구조를 정의하도록 하자 우리가 알아야하는건 주파수 모음에서의 ed, es, er값이다.(코드의 가독성을 위해서 $e_{00}$대신 물리적인 의미를 활용해 주도록 하자.)
```julia
struct OnePortCal
    freq::Vector{Float64}
    ed::Vector{ComplexF64}  # directivity(e_{00})
    es::Vector{ComplexF64}  # source match(e_{11})
    er::Vector{ComplexF64}  # reflection tracking(e_{10}e_{01})
    calset_name::Union{Nothing,String}
end
```
해당 프로젝트는 이상적인 캘킷을 가정하지 않고 실제 환경에서의 캘킷(fringing Capacitor, 기생 인덕턴스 및 딜레이, 감쇄 고려)에 맞는 캘리브레이션을 진행할 거기 때문에 내가 사용할 캘킷에 필요한 Coeff를 담아놓은 구조체가 필요하다.

```julia
struct ReflectionStandard
    label::String
    kind::Symbol
    capacitance_coeffs::NTuple{4,Float64}
    inductance_coeffs::NTuple{4,Float64}
    offset_delay_s::Float64
    offset_z0::Float64
    offset_loss_ohm_per_s::Float64이 
    terminal_z::Float64
end
```
각 open, short, load의 coefficient들을 담아놓을 구조체를 정의했다. 또한 추후에 다른 캘킷을 활용할 가능성을 염두해 두어 캘킷의 open, short, load의 coefficient를 담아놓을 구조체를 정의하자
```julia
struct OnePortCalKit
    name::String
    open::ReflectionStandard
    short::ReflectionStandard
    load::ReflectionStandard
    system_z0::Float64
end
```
여기서 주목할점은 open, short, load는 이전에 정의한 `struct ReflectionStandard`로 구조체 속의 구조체이다. 이제 내가 자주 사용하는 "Keysight 8502D" calkit의 파라미터들을 담아놓은 구조체를 생성해보자.
```julia
const CALKIT_85052D = OnePortCalKit(
    "Keysight 85052D",
    ReflectionStandard(
        "open",
        :open,
        (49.433e-15, -310.131e-27, 23.1682e-36, -0.15966e-45),
        (0.0, 0.0, 0.0, 0.0),
        29.243e-12,
        50.0,
        2.2e9,
        50.0,
    ),
    ReflectionStandard(
        "short",
        :short,
        (0.0, 0.0, 0.0, 0.0),
        (2.0765e-12, -108.54e-24, 2.1705e-33, 0.01e-42),
        31.785e-12,
        50.0,
        2.36e9,
        50.0,
    ),
    ReflectionStandard(
        "load",
        :load,
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        0.0,
        50.0,
        0.0,
        50.0,
    ),
    50.0,
)
```
다양한 캘킷 정보를 포함하고 싶으면 해당 방식보다는 csv 저장 및 load가 맞는 방식이긴 하지만... 해봤자 2~3개 정도 추가할 거 같아. 코드 내에 저장해보았다.

이제 각 coefficient들을 활용하여 Gamma 계산을 하는 코드를 작성해보자
```julia
function standard_capacitance(std::ReflectionStandard, freq_hz::Real)
    c0, c1, c2, c3 = std.capacitance_coeffs
    f = Float64(freq_hz)
    return c0 + c1 * f + c2 * f^2 + c3 * f^3
end

function standard_inductance(std::ReflectionStandard, freq_hz::Real)
    l0, l1, l2, l3 = std.inductance_coeffs
    f = Float64(freq_hz)
    return l0 + l1 * f + l2 * f^2 + l3 * f^3
end

function standard_terminal_gamma(std::ReflectionStandard, freq_hz::Real, system_z0::Real)
    omega = 2 * pi * Float64(freq_hz)
    z0 = Float64(system_z0)

    z =
        if std.kind == :open
            c = standard_capacitance(std, freq_hz)
            c == 0.0 ? ComplexF64(Inf) : 1 / (im * omega * c)
        elseif std.kind == :short
            im * omega * standard_inductance(std, freq_hz)
        elseif std.kind == :load
            ComplexF64(std.terminal_z)
        else
            error("Unsupported calibration standard kind: $(std.kind)")
        end

    isinf(real(z)) || isinf(imag(z)) ? ComplexF64(1.0, 0.0) : (z - z0) / (z + z0)
end

function offset_reflection_factor(std::ReflectionStandard, freq_hz::Real)
    phase = exp(-im * 4 * pi * Float64(freq_hz) * std.offset_delay_s)

    loss_np_oneway =
        std.offset_loss_ohm_per_s == 0.0 ? 0.0 :
        (std.offset_loss_ohm_per_s * std.offset_delay_s / (2 * std.offset_z0)) * sqrt(max(Float64(freq_hz), 0.0) / 1.0e9)

    return exp(-2 * loss_np_oneway) * phase
end

function standard_gamma(std::ReflectionStandard, freq_hz::Real, system_z0::Real)
    return standard_terminal_gamma(std, freq_hz, system_z0) * offset_reflection_factor(std, freq_hz)
end

function standard_gamma_vector(std::ReflectionStandard, freq::AbstractVector{<:Real}, system_z0::Real)
    return ComplexF64[standard_gamma(std, f, system_z0) for f in freq]
end

```

저기서 loss_np_oneway를 정의하는 방식은 다음을 따른다.
$$\alpha(f) \cdot l = \frac{\text{Loss}_{\Omega/s} \times \text{Delay}_{s}}{2 \cdot Z_{0, \text{offset}}} \times \sqrt{\frac{f}{1\text{ GHz}}}$$

감쇠와 위상 지연을 고려한 최종 반사계수는 다음과같아진다.(`standard_gamma`)
$$\Gamma_L(f) = \Gamma_{terminal}(f) \cdot e^{-2 \cdot \alpha(f)l} \cdot e^{-j \cdot 4\pi f \cdot \text{Delay}_s}$$

이제 주파수에 따른 최종 반사계수(`standard_gamma_vector`)를 활용하여 필요한 에러 항들을 구해보자

```julia
function solve_oneport_error_terms(
    freq::Vector{Float64},
    meas_open::Vector{ComplexF64},
    meas_short::Vector{ComplexF64},
    meas_load::Vector{ComplexF64};
    calkit::OnePortCalKit = CALKIT_85052D,
    calset_name::Union{Nothing,AbstractString} = nothing,
)
    n = length(freq)
    length(meas_open) == n || error("OPEN measurement length does not match frequency length")
    length(meas_short) == n || error("SHORT measurement length does not match frequency length")
    length(meas_load) == n || error("LOAD measurement length does not match frequency length")

    gamma_open = standard_gamma_vector(calkit.open, freq, calkit.system_z0)
    gamma_short = standard_gamma_vector(calkit.short, freq, calkit.system_z0)
    gamma_load = standard_gamma_vector(calkit.load, freq, calkit.system_z0)

    ed = Vector{ComplexF64}(undef, n)
    es = Vector{ComplexF64}(undef, n)
    er = Vector{ComplexF64}(undef, n)

    for i in 1:n
        g = (gamma_open[i], gamma_short[i], gamma_load[i])
        m = (meas_open[i], meas_short[i], meas_load[i])
        a = ComplexF64[
            1.0 g[1] * m[1] g[1]
            1.0 g[2] * m[2] g[2]
            1.0 g[3] * m[3] g[3]
        ]
        x = a \ ComplexF64[m[1], m[2], m[3]] # 역행렬로 선형 방정식 풀기
        ed[i] = x[1]
        es[i] = x[2]
        er[i] = x[3] + x[1] * x[2]
    end

    return OnePortCal(freq, ed, es, er, isnothing(calset_name) ? nothing : String(calset_name))
end
```

$$\Gamma_{m} = e_d + \frac{e_r \Gamma_L}{1 - e_s \Gamma_L}$$ 

1-port 교정에서 보았던 선형 시스템을 선형 방정식 형태로 바꾸면 다음과 같이 된다.
$$\Gamma_m = e_d + \Gamma_m \Gamma_L e_s + (e_r - e_d e_s) \Gamma_L$$

- `x[1]`: $e_d$ (지향성)
- `x[2]`: $e_s$ (소스 매치)
- `x[3]`: $\Delta = e_r - e_d e_s$ (중간 변수)

와 같이 치환을 해주면 
$$A = \begin{bmatrix} 1 & \Gamma_{L,1} \Gamma_{m,1} & \Gamma_{L,1} \\ 1 & \Gamma_{L,2} \Gamma_{m,2} & \Gamma_{L,2} \\ 1 & \Gamma_{L,3} \Gamma_{m,3} & \Gamma_{L,3} \end{bmatrix}, \quad B = \begin{bmatrix} \Gamma_{m,1} \\ \Gamma_{m,2} \\ \Gamma_{m,3} \end{bmatrix}$$ 다음과 같이 선형 방정식을 행렬로 구성할 수 있다. `x[1], x[2], x[3]`는 `A \ B`(역행렬 구하기)가 된다. 

이렇게 나온 `OnePortCal` 객체를 활용해 캘리브레이션을 진행하
```julia
function apply_oneport_cal(s11_raw::Vector{ComplexF64}, cal::OnePortCal)
    
    numerator = s11_raw - cal.ed
    denominator = cal.es .* s11_raw + cal.er - cal.ed .* cal.es
    return numerator ./ denominator
end
```
**보정 방정식**
$$S_{11, cal} = \frac{S_{11, raw} - e_d}{e_s S_{11, raw} + (e_r - e_d e_s)}$$

1. 이항: $\Gamma_m - e_d = \frac{e_r \Gamma_a}{1 - e_s \Gamma_a}$
2. 역수 및 정리: $(\Gamma_m - e_d)(1 - e_s \Gamma_a) = e_r \Gamma_a$
3. $\Gamma_a$에 대해 묶기: $(\Gamma_m - e_d) - e_s \Gamma_a (\Gamma_m - e_d) = e_r \Gamma_a$
4. 최종 정리: $\Gamma_a = \frac{\Gamma_m - e_d}{e_r + e_s(\Gamma_m - e_d)}$

이거로 캘리브레이션에 대한 준비는 마쳤다. 사실 VNA 자체에서 캘리브레이션된 데이터를 불러오는 방식도 있다... 훨씬 편하지만 스위치를 사용할 때는 캘 데이터 4개가 필요하기 때문에 VNA 자체에서는 제약이 있을수 있다.