// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "apple-speech-worker",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(name: "AppleSpeechWorkerCore", targets: ["AppleSpeechWorkerCore"]),
        .executable(name: "apple-speech-worker", targets: ["apple-speech-worker"]),
        .executable(
            name: "apple-speech-worker-contract-tests",
            targets: ["AppleSpeechWorkerContractTests"]
        )
    ],
    targets: [
        .target(name: "AppleSpeechWorkerCore"),
        .executableTarget(
            name: "apple-speech-worker",
            dependencies: ["AppleSpeechWorkerCore"]
        ),
        .executableTarget(
            name: "AppleSpeechWorkerContractTests",
            dependencies: ["AppleSpeechWorkerCore"]
        )
    ]
)
