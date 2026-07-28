// SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
//
// SPDX-License-Identifier: MIT

#include "CameraStream.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Modules/ModuleManager.h"

// Sets default values
ACameraStream::ACameraStream()
{
    // Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
    PrimaryActorTick.bCanEverTick = true;
    Capture = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("Capture"));
    RootComponent = Capture;
}

// Called when the game starts or when spawned
void ACameraStream::BeginPlay()
{
    Super::BeginPlay();

    RT = NewObject<UTextureRenderTarget2D>(this);
    RT->InitAutoFormat(Width, Height);
    RT->RenderTargetFormat = RTF_RGBA8;
    RT->UpdateResourceImmediate(true);

    Capture->TextureTarget = RT;
    Capture->CaptureSource = SCS_FinalColorLDR;

    Connect();
}

bool ACameraStream::Connect()
{
    Socket = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)
                ->CreateSocket(NAME_Stream, TEXT("CamStream"), false);
    TSharedRef<FInternetAddr> Addr =
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateInternetAddr();
    bool bValid;
    Addr->SetIp(*ServerIP, bValid);
    Addr->SetPort(ServerPort);
    return Socket && Socket->Connect(*Addr);
}

void ACameraStream::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
    Timer += DeltaTime;
    if (Timer < 1.f / TargetFPS)
        return;
    Timer = 0.f;
    if (Socket && Socket->GetConnectionState() == SCS_Connected)
        SendFrame();
    else
        Connect(); // reconnect
}

void ACameraStream::SendFrame()
{
    TArray<FColor> Pixels;
    RT->GameThread_GetRenderTargetResource()->ReadPixels(Pixels);

    // BGRA → RGBA
    for (FColor &C : Pixels)
        Swap(C.R, C.B);

    IImageWrapperModule &IWM =
        FModuleManager::LoadModuleChecked<IImageWrapperModule>("ImageWrapper");
    TSharedPtr<IImageWrapper> Wrapper = IWM.CreateImageWrapper(EImageFormat::JPEG);
    Wrapper->SetRaw(Pixels.GetData(), Pixels.Num() * sizeof(FColor),
                    Width, Height, ERGBFormat::RGBA, 8);
    TArray64<uint8> Jpeg = Wrapper->GetCompressed(80);

    // 4-Byte Header: Länge des JPEG
    uint32 Len = (uint32)Jpeg.Num();
    uint8 Header[4] = {uint8(Len >> 24), uint8(Len >> 16), uint8(Len >> 8), uint8(Len)};

    int32 Sent = 0;
    Socket->Send(Header, 4, Sent);
    Socket->Send(Jpeg.GetData(), Jpeg.Num(), Sent);
}

void ACameraStream::EndPlay(EEndPlayReason::Type R)
{
    Super::EndPlay(R);
    if (Socket)
    {
        Socket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(Socket);
    }
}
