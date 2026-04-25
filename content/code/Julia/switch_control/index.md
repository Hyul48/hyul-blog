---
title: "Julia Switch Controller"
description: "Julia Switch Controller"
weight: 1
math: true
---
# Switch control

이번 포스트는 줄리아를 이용해 RF 스위치를 제어하는 방법에 대해 다뤄보도록한다.

사용한 장비는 NI USB-6501(디지털 I/O)을 활용해 mini circuit의 TTL-1S4PT(스위치)이다.

제어 다이어 그램은 다음과 같다.

![이미지1](diagram2.png)

전자공학 기초 실험을 접해본 분들이라면 '디지털 신호에 의한 제어'라는 개념이 익숙하실 겁니다. 우리가 흔히 사용하는 RF 스위치 제어 시스템도 알고 보면 이 기초적인 원리를 따릅니다.

컴퓨터 내부의 데이터는 본질적으로 '0'과 '1'의 조합입니다. 하지만 컴퓨터 자체로는 실제 하드웨어에 전기적 신호를 보낼 수 없습니다. 이때 NI USB-6501과 같은 Digital I/O 모듈이 가교 역할을 합니다.

사용자가 제어 소프트웨어에서 특정 명령을 내리면 USB-6501은 이를 받아 미리 약속된 디지털 값을 실제 물리적인 전압으로 변환하여 내보냅니다.

제어 모듈에서 나온 전압신호는 TTL-1S4PT의 제어핀으로 입력됩니다. 스위치 내부에는 이 입력된 전압의 조합을 해석하는 '디코더' 회로가 들어있습니다. 예를 들어 세개의 제어 선에 각각 Low-Low-High라는 신호가 들어오면 스위치 내부에서는 2번 포트와 COM포트를 연결하라는 논리로 해석합니다. 이 해석 결과에 따라 내부 소자가 동작하여 RF 경로가 물리적을 연결되는 원리입니다.

먼저 필자의 코딩 스타일은 적당한 모듈화를 선호하기 때문에(추후에 재활용성이 높아짐) 스위치 제어 모듈을 Julia로 작성하는 방법에 대해 다뤄보겠습니다.

```julia
module SwitchMIMOControl
```
먼저 모듈을 선언해 줍니다. 실제로 통합코드에서는 switchMIMOControl 모듈을 불러와서 사용해주면 됩니다.

```julia
export RFPort, PORT_1, PORT_2, PORT_3, PORT_4
export AbstractSwitchBackend, MockSwitchBackend, FunctionSwitchBackend, NIDaqSwitchBackend
export SwitchController, set_switch_state!, select_port!, all_off!, active_port
```
실제 모듈에서 선언된 함수들을 밖에서 불러오기 위해선 export로 선언을 해줘야 합니다.

```julia
@enum RFPort::Int begin
    PORT_1 = 1
    PORT_2 = 2
    PORT_3 = 3
    PORT_4 = 4
end
```
먼저 코드의 가독성을 높이기 위해 1,2,3,4로 선언되어 있는 port를 PORT_1, PORT_2,...로 변경해 줍니다.

굳이 이렇게 해주는 이유는
```julia
select_port!(controller, PORT_1)
```
이 코드가
```julia
select_port!(controller, 1)
```
이 코드보다 의미가 명확하기 때문입니다.(1이 포트를 의미하는지 숫자 1을 의미하는지 알 수 없음)

## 추상 타입으로 백엔드 계층 만들기
제가 생각하는 ```julia``` 언어의 장점은 추상 타입 + 다중 디스패치입니다. 

**추상 타입 : 유연한 계층 구조의 설계도**
```Juila```에서 추상 타입은 실제 값을 가지지는 않지만, 다른 타입들을 묶어주는 카테고리 역할을 합니다. 예를 들어 ```AbstractAntenna```나 ```AbstractSensor``` 같은 타입을 정의해두면, 나중에 새로운 장비가 추가되어도 기존 로직을 추가할 필요가 없습니다.

**다중 디스패치 : 상황에 맞는 가장 똑똑한 함수 호출**
```Julia```의 꽃이라고 불리는 다중 디스패치는 함수의 인자로 들어온 모든 타입과 조합을 보고 가장 적합한 함수 버전을 실행하는 방식입니다.
- 객체지향(Python, C++ 등) : 보통의 OOP는 '메서드'가 객체에 속해 있어 ```object.method()``` 형태를 띱니다. 반면 ```Julia```는 ```method(arg1, arg2)``` 형태이며, arg1과 arg2에 따라 실행될 코드가 결정됩니다.
- 실제 동작 : ```transmit(antenna, siganl)```이라는 함수가 있을 때 ```type(antenna) = horn```인지 ```type(antenna) = patch```인지에 따라 최적화된 연산을 자동으로 수행합니다.

**Comapare Python vs Julia**
```python
# python 코드
class PatchAntenna:
    def transmit(self, signal):
        print(f"Patch Antenna: Transmitting {signal}")

class HornAntenna:
    def transmit(self, signal):
        print(f"Horn Antenna: Transmitting {signal}")

# 호출 방식
antenna = PatchAntenna()
antenna.transmit("5G Signal") # 주체인 antenna가 무엇인지에 따라 결정됨
```

```Julia
# 줄리아 코드
abstract type AbstractAntenna end
struct PatchAntenna <: AbstractAntenna end
struct HornAntenna <: AbstractAntenna end

abstract type AbstractSignal end
struct DigitalSignal <: AbstractSignal end
struct AnalogSignal <: AbstractSignal end

# 함수가 객체 밖에서 정의됨 (다양한 조합)
transmit(ant::PatchAntenna, sig::DigitalSignal) = println("Patch + Digital: High-speed switching")
transmit(ant::PatchAntenna, sig::AnalogSignal)  = println("Patch + Analog: Basic modulation")
transmit(ant::HornAntenna,  sig::DigitalSignal) = println("Horn + Digital: High-gain directional")

# 호출 방식
ant = PatchAntenna()
sig = DigitalSignal()

transmit(ant, sig) # 인자 두 개(ant, sig)의 타입을 모두 고려해 최적의 함수를 선택
```

따라서 스위치를의 백엔드를 2개로 쪼개봅시다.
```julia
abstract type AbstractSwitchBackend end
```
Backend에 대한 추상객체를 정의해줍니다.


| 백엔드                     | 역할                         |
| ----------------------- | -------------------------- |
| `MockSwitchBackend`     | 실제 장비 없이 명령 로그만 기록         |
| `NIDaqSwitchBackend`    | NI-DAQ 장비를 통해 실제 디지털 출력 수행 |

```julia
mutable struct MockSwitchBackend <: AbstractSwitchBackend
    log::Vector{NamedTuple{(:timestamp, :command), Tuple{DateTime, Any}}}
end
```
해당 벡엔드는 실제 장비를 건드리지 않고, 어떤 명령이 내려졌는지만 기록합니다.

```julia
mutable struct NIDaqSwitchBackend <: AbstractSwitchBackend
    task_handle::Ptr{Cvoid}
    line_spec::String
    drive_type::Union{Nothing,Int32}
    started::Bool
    log::Vector{NamedTuple{(:timestamp, :command), Tuple{DateTime, Any}}}
end
```
해당 벡엔드는 실제로 장비를 컨트롤 하는 명령을 전달합니다. 하드웨어 컨트롤 코드를 짤때는 Mock 버전의 코드를 작성하는 것도 하나의 좋은 방법입니다.

## SwitchController : 전체 상태 관리 객체
```julia
mutable struct SwitchController{B<:AbstractSwitchBackend,C}
    backend::B
    port_commands::Dict{RFPort,C}
    off_command::C
    settle_time_s::Float64
    active::Union{Nothing,RFPort}
end
```
먼저 매개 변수가 다음과 같이 정의되어있습니다.
```B<:AbstractSwitchBackend,C``` 의미는 B라는 매개 변수와 C라는 매개변수를 받는데 B는 Backend 객체가 들어와야하고 C는 아직 타입에 대한 정의가 없습니다.

이 구조체는 RF 스위치 제어에 필요한 모든 상태를 담고 있습니다.
| 필드              | 의미                |
| --------------- | ----------------- |
| `backend`       | 실제 명령을 처리할 백엔드    |
| `port_commands` | 각 포트에 대응하는 명령     |
| `off_command`   | 모든 포트를 끄는 명령      |
| `settle_time_s` | 포트 전환 후 안정화 대기 시간 |
| `active`        | 현재 선택된 포트         |

```Dict, Float```는 아마 다들 아시겠지만 ```Union```은 익숙하지 않을 수 있습니다. 
```Union```은 "정수이거나 문자열"이라는 뜻을 가집니다. 좀 더 엄밀히 따져보자면... Julia에는 Any를 선언할 수 있습니다. 내가 타입을 모를 때 Any를 선언하면 줄리아가 후보군들을 파악하여 최적화를 진행하는데 이 때 후보군을 좁혀 코드의 속도를 빠르게 해주는 역할을 합니다. 

**컨트롤러 생성자**
생성자는 구조(Struct)를 생성 및 설정하는 함수를 의미합니다.
```julia
function SwitchController(
    backend::B;
    port_commands::AbstractDict = Dict(
        PORT_1 => :RF1_ON,
        PORT_2 => :RF2_ON,
        PORT_3 => :RF3_ON,
        PORT_4 => :RF4_ON,
    ),
    off_command = :RF_OFF,
    settle_time_s::Real = 0.1,
) where {B<:AbstractSwitchBackend}
    typed_commands = Dict{RFPort, typeof(off_command)}()
    for port in instances(RFPort)
        haskey(port_commands, port) || error("Missing switch command for $port")
        typed_commands[port] = convert(typeof(off_command), port_commands[port])
    end

    return SwitchController(
        backend,
        typed_commands,
        off_command,
        Float64(settle_time_s),
        nothing,
    )
end
```
위 코드는 SwitchContoller라는 객체를 처음 만들 때 활용하는 생성자 코드입니다.
내가 ```PORT_1```이라는 신호를 보내면 RF1_ON이라는 신호를 보내라고 약속하는 Dict를 정의합니다.
```julia
for port in instances(RFPort)
    haskey(port_commands, port) || error("Missing switch command for $port")
end
```
는 ```port_commands``` 안에 이 4개의 명령어가 전부 들어있는지 확인합니다. 일종의 안정성 검사입니다.
이후 Switchcontroller 객체를 반환하는 것을 확인할 수 있습니다.

**디지털 출력 패턴**
```julia
default_port_commands() = Dict(
    PORT_1 => UInt8[1, 1, 0],
    PORT_2 => UInt8[0, 1, 0],
    PORT_3 => UInt8[1, 0, 0],
    PORT_4 => UInt8[0, 0, 0],
)
``` 
파이썬에도 함수를 간략하게 선언하는 방법이 있듯이 julia도 단축형 함수 정의가 있습니다. 함수 ```default_port_commands()```를 호출하면 Dict를 반환해주는 역할을 합니다. Dict는 PORT_#(RF#_ON) 명령어가 어떤 디지털 코드를 가지는지 정리되어 있습니다. 

여기까지 따라오신 분들은 굳이 코드를 이렇게 어렵게 짜는 이유에 대해 궁금하실 수 있습니다. "그냥 RF1 킬거면 ```UInt8[1, 1, 0]``` 보내도록 작성하면 되는거 아닐까?"라고 새각할 수 있는데 사실 맞습니다... ㅎㅎ 굳이 이렇게 짠 이우는 지금 제 스위치는 ```UInt8[1, 1, 0]```이 RF1_ON 신호일 수도 있지만 다른 스위치는 그렇지 않을 수도 있죠... 그래서 코드의 재활용성을 높이는 방안이라고 보시면 될 거 같습니다. 예를 들어 스위치 2의 제어신호가 다르다면 다음과 같은 정의를 추가 정의할 수 있겠죠

```julia
switch2_port_commands() = Dict(
    PORT_1 => UInt8[0, 0, 1],
    PORT_2 => UInt8[0, 1, 0],
    PORT_3 => UInt8[1, 0, 0],
    PORT_4 => UInt8[0, 1, 1],
)
``` 
스위치가 아니여도 디코더로 동작하는 녀석들이라면 모두 가능할겁니다.

**NI-DAQ 연동: Julia의 ```ccall```**
해당 부분이 하드웨어 연동의 핵심입니다. 단순히 usb를 꽂는다고 julia가 알아서 usb가 꽂혔으니깐 이 장비를 조작하면 되겠구나!라고 생각하진 않을겁니다. 따라서 내가 제어할 장비에 대해서 선을이 필요합니다.

```julia
const NIDAQMX_LIB = "nicaiu.dll"

status = ccall(
    (:DAQmxGetSysDevNames, NIDAQMX_LIB),
    Int32,
    (Ptr{UInt8}, UInt32),
    buf,
    UInt32(length(buf))
)
```

```ccall```함수는 C 라이브러리 함수를 직접 호출할 수 있게 해줍니다. 여기서는 ```nicaiu.dll```, 즉 NI-DAQmx 드라이버의 DLL을 호출합니다.



사실 장비제어에서 ```ccall```은 ```Julia```가 ```python```보다 이점을 가질 수 있는 이유중 하나입니다.
```Julia```는 별도의 라이브러리 작성 없이 ```ccall``` 키워드를 통해 C 라이브러리 함수를 직접 호출할 수 있습니다. 따라서 거의 C 속도에 근접할 수 있습니다. Python 역시 C API가 존재하지만 인터프리터 언어의 특성때문에(C가 이해하려면 추가 번역이 필요합니다.) 성능이 떨어집니다.(사실 이 과정은 여러번 호출할 때 의미가 있습니다.) 

아래는 ```nicaiu.dll``` 라이브러리에 정의된 함수들입니다.

| 함수명 | 용도 (Purpose) | 주요 매개변수 (C 기준) | Julia 대응 타입 (`ccall`용) |
|---|---|---|---|
| `DAQmxGetExtendedErrorInfo` | 마지막으로 발생한 상세 에러 메시지 조회 | `char errorString[], uInt32 bufferSize` | `(Ptr{UInt8}, UInt32)` |
| `DAQmxGetSysDevNames` | 시스템에 연결된 모든 NI 장비 이름 조회 | `char data[], uInt32 bufferSize` | `(Ptr{UInt8}, UInt32)` |
| `DAQmxCreateTask` | 새로운 작업(Task) 생성 | `const char taskName[], TaskHandle *taskHandle` | `(Cstring, Ptr{Ptr{Nothing}})` |
| `DAQmxStartTask` | 설정된 작업 시작 | `TaskHandle taskHandle` | `(Ptr{Nothing},)` |
| `DAQmxStopTask` | 실행 중인 작업 중지 | `TaskHandle taskHandle` | `(Ptr{Nothing},)` |
| `DAQmxClearTask` | 작업 삭제 및 리소스 해제 | `TaskHandle taskHandle` | `(Ptr{Nothing},)` |
| `DAQmxGetErrorString` | 에러 코드(숫자)를 기본 메시지로 변환 | `int32 errorCode, char errorString[], uInt32 bufferSize` | `(Int32, Ptr{UInt8}, UInt32)` |
|`DAQmxWriteDigitalLines`|디지털 라인에 데이터(전압) 출력|`taskHandle, numSamps, autoStart, timeout, layout, writeArray, sampsWritten, reserved`|`(Ptr{Cvoid}, Int32, UInt32, Float64, UInt32, Ptr{UInt8}, Ref{Int32}, Ptr{Cvoid})`|

이제 julia에서 라이브러리를 이용해 필요한 함수를 만들어줍니다.

**에러 판별(장치 연결)**
```julia
function nidaqmx_check(code::Integer, msg::AbstractString)
    code < 0 || return nothing

    errbuf = Vector{UInt8}(undef, 2048)
    ccall((:DAQmxGetExtendedErrorInfo, NIDAQMX_LIB), Int32, (Ptr{UInt8}, UInt32), errbuf, UInt32(length(errbuf)))
    terminator = findfirst(==(0x00), errbuf)
    detail = terminator === nothing ? String(errbuf) : String(errbuf[1:terminator-1])
    error("NI-DAQmx error $code while $msg: $detail")
end
```

DAQ에서 보낸 에러 메세지를 받아오는 명령어입니다. ```C```는 항상 문장의 끝에 ```0x00```을 붙여주기 때문에 이를 이용하여 종결차를 찾아 실제로 끝나는 지점까지만 자르는게 가능합니다. 물론 ```0x00```으로 안끝날 수도 있으니 그 부분에 대한 예외 처리가 필요합니다.(```detail = terminator === nothing ? String(errbuf) : String(errbuf[1:terminator-1])```) 이 코드를 이해하기 위해선 실제로 활용되는 것을 보는게 도움이 됩니다.

```julia
function list_nidaq_devices()
    buf = Vector{UInt8}(undef, 4096)
    status = ccall((:DAQmxGetSysDevNames, NIDAQMX_LIB), Int32, (Ptr{UInt8}, UInt32), buf, UInt32(length(buf)))
    nidaqmx_check(status, "listing NI-DAQ devices")
    terminator = findfirst(==(0x00), buf)
    text = terminator === nothing ? String(buf) : String(buf[1:terminator-1])
    return filter(!isempty, strip.(split(text, ',')))
end
```
위의 함수는 연결된 장비를 조회하는 코드인데 연결된 장비를 찾고 code가 0보다 작으면 이 때 에러가 발생하면 에러 메세지를 출력합니다. 


## 디지털 전압 출력을 위한 함수 구현 : 다중 디스패치
```julia
function set_switch_state!(backend::MockSwitchBackend, command)
    push!(backend.log, (timestamp = now(), command = command))
    return command
end

function set_switch_state!(backend::NIDaqSwitchBackend, command)
    # 데이터 타입에 대한 엄밀한 검사 진행
    bytes = command isa AbstractVector ? UInt8.(collect(command)) : throw(ArgumentError("NI-DAQ backend expects a vector command, got $(typeof(command))"))
    samps_written = Ref{Int32}(0)
    data = Vector{UInt8}(bytes)

    
    status = ccall(
        (:DAQmxWriteDigitalLines, NIDAQMX_LIB),
        Int32,
        (Ptr{Cvoid}, Int32, UInt32, Float64, UInt32, Ptr{UInt8}, Ref{Int32}, Ptr{Cvoid}),
        backend.task_handle, # 제어할 장비
        1, 
        1,
        10.0,
        DAQMX_VAL_GROUP_BY_CHANNEL,
        pointer(data), # 포인터로 직접 넘기기 때문에 C 언어와 같은 속도로 동작
        samps_written,
        C_NULL,
    )
    nidaqmx_check(status, "writing switch command $(collect(data))")
    backend.started = true
    push!(backend.log, (timestamp = now(), command = copy(data)))
    return copy(data)
end
```
위는 디지털 전압 출력을 위한 함수를 정의합니다. 백엔드의 종류에 따라 2개의 함수를 구현합니다. 
```MockSwitchBackend```의 : 경우 단순히 log에 무슨 명령어를 추가했는지 확인합니다.
```NIDaqSwitchBackend```의 경우 :
먼저 `C`는 Julia보다 훨씬 데이터 타입에 엄격합니다. 데이터 타입 검사가 엄밀하게 진행되어야 합니다.

위의 코드에서 `ccall`을 통해 불러온 함수의 인자들을 살펴보면 다음과 같습니다.
| 인자 위치 | 값/변수 | 설명 (Physical Meaning) |
|---|---|---|
| 1st | `backend.task_handle` | 제어할 장비(USB-6501)의 고유 식별자, 즉 핸들 |
| 2nd | `1` | 각 채널당 쓸 데이터 샘플 수임. 단발성 스위칭이라 1개만 씀. |
| 3rd | `1` | `autoStart` 옵션임. `1(True)`로 설정해서 호출 즉시 전압이 바뀌게 함.(즉시 시작 옵션) |
| 4th | `10.0` | 언제까지? 타임아웃 시간(초)임. 10초 안에 명령이 안 들어가면 에러를 냄. |
| 5th | `DAQMX_VAL_GROUP_BY_CHANNEL` | 어떤 순서로? 데이터를 채널별로 묶어서 배치하겠다는 설정임. |
| 6th | `pointer(data)` | 실제 신호(`[1, 1, 0]`)가 담긴 메모리 주소를 넘겨줌. 직통 통로의 핵심임. |
| 7th | `samps_written` | 확인용 : “실제로 몇 개가 써졌니?”를 장비에 되묻기 위해 `Ref` 주소를 넘겨줌. |
| 8th | `C_NULL` | 예약 공간 : NI 드라이버 규격상 현재는 쓰지 않는 빈 자리임. |


## PORT 선택
이제 선택된 포트로 데이터 전압을 출력해봅시다.

```julia
function select_port!(controller::SwitchController, port::RFPort; settle::Bool = true)
    command = controller.port_commands[port]
    set_switch_state!(controller, command)
    controller.active = port
    settle && sleep(controller.settle_time_s)
    return port
end
```
동작 순서는 간단합니다. 매개변수로 port를 받으면 port에 해당하는 명령을 찾습니다. 이후 해당 명령을 백엔드로 보내고 현재 활성 포트 상태를 업데이트 합니다. 또한 데이터 입출력 및 장비제어 시간이 필요하므로 안정화 시간을 기다린 후 선택된 포트를 return해 줍니다. `julia`에선 관례적으로 객체의 상태를 변환하는 함수의 이름에는 !를 쓴다는 것도 알아두면 좋습니다.

## 모든 포트 끄기
```julia
function all_off!(controller::SwitchController; settle::Bool = true)
    set_switch_state!(controller, controller.off_command)
    controller.active = nothing
    settle && sleep(controller.settle_time_s)
    return nothing
end
```
PORT 선택과 메카니즘은 거의 동일합니다. 어떻게 동작하는지는 PORT 선택을 참고하시면 됩니다.

## Cycle_PORTS : 순환 측정을 위한 제어
```julia
function cycle_ports!(
    f::Function,
    controller::SwitchController;
    ports = instances(RFPort),
    settle::Bool = true,
    switch_off_at_end::Bool = false,
)
    port_list = collect(ports)
    results = Vector{Any}(undef, length(port_list))
    try
        for (i, port) in pairs(port_list)
            rf_port = port isa RFPort ? port : RFPort(port)
            select_port!(controller, rf_port; settle = settle)
            results[i] = f(rf_port)
        end
    catch
        if switch_off_at_end
            try
                all_off!(controller; settle = settle)
            catch err
                @warn "Failed to switch all ports off after port-cycle error" exception = err
            end
        end
        rethrow()
    end
    if switch_off_at_end
        all_off!(controller; settle = settle)
    end
    return results
end
```
저에게 필요한 작업이 포트를 순회하면서 데이터를 여러 프레임 측정해야 하기때문에 port를 순회하는 코드를 정의해줍시다. 즉 `RF1측정` $\rightarrow$ ... $\rightarrow$ `RF4측정`을 한 개의 함수로 할 수 있도록 정의합니다. `f::Function`을 보면 아시겠지만 함수를 그때 그때 호출하여 사용합니다. 예를들면 `function = on`을 쓰면 순환하면서 on하는 함수를 호출하고 `function = off`를 하면 순환하면서 off하는 함수를 호출합니다.(사실 이 프로젝트와는 큰 상관이 없는 예시입니다.) 컴파일러 언어의 특성은 해당 작업이 심한 오버헤드를 발생시키지 않는다는 장점이 있습니다. 이렇게 코드를 작성했을 때 장점은 다음과 같은 응용이 가능해집니다.

```julia
SwitchMIMOControl.cycle_ports!(controller; ports = 1:N_SWITCH_PORTS, switch_off_at_end = true) do port
        port_num = Int(port)
        println("\n[$pass_name] Selecting switch port $port_num...")
        cal = cal_by_port[port]
        activate_calset(instr, cal)
        println("[$pass_name] Acquiring VNA trace at port $port_num...")

        freq, s11_raw, s11_cal = acquire_calibrated_trace(instr, cal; n_avg = n_avg)
        point_dir = joinpath(pass_dir, port_dirname(port))
        mkpath(point_dir)
        save_live_outputs(point_dir, freq, s11_raw, s11_cal; save_snapshot = true)
        print_live_summary(port_num, freq, s11_raw, s11_cal)
        push!(point_dirs, point_dir)
        return nothing
    end
```

port를 순환하면서 
```julia
do port
    port_num = Int(port)
    println("\n[$pass_name] Selecting switch port $port_num...")
    cal = cal_by_port[port]
    activate_calset(instr, cal)
    
    # ... (VNA 데이터 측정 및 저장 로직들) ...
    
    push!(point_dirs, point_dir)
    return nothing
end
```
와 같은 동작들이 함수처럼 정의가 되면서 통채로 `f`처럼 작동하게 할 수 있습니다.
간단하게 말해서 

> "내가 순환하면서 어떤 동작을 할진 모르겠는데 그건 나중에 정의할 테니깐 포트를 순환하면서 어떤 동작을 하는 로직을 정의해줘. 어떤 동작은 `do`와 `end`사이에 정의될거야"

해당 로직은 굉장히 강력한 무기가 됩니다. 예를 들어 봅시다.
RF에서 측정을 위해선 calibration, background측정, target측정이 필요합니다. 이 3가지 과정 모두 `cycle_port`가 필요합니다. 그럼 3가지 과정은 순차적으로 다음과 같이 정의가 됩니다.

```julia
cycle_port!(...) do port (캘리브레이션) end
cycle_port!(...) do port (백그라운드 측정) end
cycle_port!(...) do port (타겟 측정) end
```

이렇게 줄리아 스위치 제어 모듈이 완성됐습니다... 사실 이 코드를 작성하면서 G선생님의 도움이 너무 컸지만 막상 정리를 하다보니 다시금 `JULIA` 언어의 매력에 빠지게 되네요... 그럼 이만(공격적인 피드백 적극 수용)