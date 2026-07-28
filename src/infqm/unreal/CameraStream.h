// SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
//
// SPDX-License-Identifier: MIT

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "CameraStream.generated.h"

UCLASS()
class MYPROJECT_API ACameraStream : public AActor
{
    GENERATED_BODY()

public:
    // Sets default values for this actor's properties
    ACameraStream();

    UPROPERTY(EditAnywhere, Category = "Stream")
    int32 Width = 640;
    UPROPERTY(EditAnywhere, Category = "Stream")
    int32 Height = 480;
    UPROPERTY(EditAnywhere, Category = "Stream")
    FString ServerIP = "127.0.0.1";
    UPROPERTY(EditAnywhere, Category = "Stream")
    int32 ServerPort = 9870;
    UPROPERTY(EditAnywhere, Category = "Stream")
    float TargetFPS = 10.f;

protected:
    // Called when the game starts or when spawned
    void BeginPlay() override;
    void EndPlay(EEndPlayReason::Type) override;
    void Tick(float DeltaTime) override;

private:
    USceneCaptureComponent2D *Capture = nullptr;
    UTextureRenderTarget2D *RT = nullptr;
    FSocket *Socket = nullptr;
    float Timer = 0.f;

    bool Connect();
    void SendFrame();
};
