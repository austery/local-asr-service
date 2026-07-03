import Foundation

public protocol AppleSpeechRuntime: AnyObject, Sendable {
    func capabilities() async throws -> WorkerCapabilities
    func prepareAsset(_ request: AssetPreparationRequest) async throws -> AssetPreparationResult
    func transcribe(_ request: TranscriptionRequest) async throws -> TranscriptionResult
}

public enum WorkerError: Error, Equatable, Sendable, CustomStringConvertible {
    case invalidArguments([String])
    case missingFakeResult(String)
    case unsupportedModule(WorkerSpeechModule)

    public var description: String {
        switch self {
        case let .invalidArguments(arguments):
            "invalid arguments: \(arguments.joined(separator: " "))"
        case let .missingFakeResult(name):
            "missing fake runtime result: \(name)"
        case let .unsupportedModule(module):
            "unsupported module: \(module.rawValue)"
        }
    }
}

public final class FakeAppleSpeechRuntime: AppleSpeechRuntime, @unchecked Sendable {
    public var capabilitiesResult: WorkerCapabilities?
    public var assetPreparationResult: AssetPreparationResult?
    public var transcriptionResult: TranscriptionResult?

    public init() {}

    public func capabilities() async throws -> WorkerCapabilities {
        guard let capabilitiesResult else {
            throw WorkerError.missingFakeResult("capabilities")
        }
        return capabilitiesResult
    }

    public func prepareAsset(_ request: AssetPreparationRequest) async throws -> AssetPreparationResult {
        guard let assetPreparationResult else {
            throw WorkerError.missingFakeResult("prepareAsset")
        }
        return assetPreparationResult
    }

    public func transcribe(_ request: TranscriptionRequest) async throws -> TranscriptionResult {
        guard let transcriptionResult else {
            throw WorkerError.missingFakeResult("transcribe")
        }
        return transcriptionResult
    }
}
