import Foundation

public enum WorkerCommand: Equatable, Sendable {
    case capabilities
    case prepare(AssetPreparationRequest)
    case transcribe(TranscriptionRequest)

    public static func parse(arguments: [String]) throws -> WorkerCommand {
        if arguments == ["capabilities"] || arguments == ["capabilities", "--json"] {
            return .capabilities
        }
        if arguments.first == "prepare" {
            return try parsePrepare(arguments: arguments)
        }
        if arguments.first == "transcribe" {
            return try parseTranscribe(arguments: arguments)
        }
        throw WorkerError.invalidArguments(arguments)
    }

    private static func parsePrepare(arguments: [String]) throws -> WorkerCommand {
        var locale: String?
        var module: WorkerSpeechModule?
        var index = 1

        while index < arguments.count {
            switch arguments[index] {
            case "--locale":
                guard index + 1 < arguments.count else {
                    throw WorkerError.invalidArguments(arguments)
                }
                locale = arguments[index + 1]
                index += 2
            case "--module":
                guard index + 1 < arguments.count,
                      let parsedModule = WorkerSpeechModule(rawValue: arguments[index + 1])
                else {
                    throw WorkerError.invalidArguments(arguments)
                }
                module = parsedModule
                index += 2
            case "--json":
                index += 1
            default:
                throw WorkerError.invalidArguments(arguments)
            }
        }

        guard let locale, let module else {
            throw WorkerError.invalidArguments(arguments)
        }
        return .prepare(AssetPreparationRequest(locale: locale, module: module))
    }

    private static func parseTranscribe(arguments: [String]) throws -> WorkerCommand {
        var inputPath: String?
        var locale: String?
        var module: WorkerSpeechModule?
        var audioTimeRanges = true
        var includeVolatile = false
        var index = 1

        while index < arguments.count {
            switch arguments[index] {
            case "--input":
                guard index + 1 < arguments.count else {
                    throw WorkerError.invalidArguments(arguments)
                }
                inputPath = arguments[index + 1]
                index += 2
            case "--locale":
                guard index + 1 < arguments.count else {
                    throw WorkerError.invalidArguments(arguments)
                }
                locale = arguments[index + 1]
                index += 2
            case "--module":
                guard index + 1 < arguments.count,
                      let parsedModule = WorkerSpeechModule(rawValue: arguments[index + 1])
                else {
                    throw WorkerError.invalidArguments(arguments)
                }
                module = parsedModule
                index += 2
            case "--audio-time-ranges":
                guard index + 1 < arguments.count,
                      let parsedValue = parseBool(arguments[index + 1])
                else {
                    throw WorkerError.invalidArguments(arguments)
                }
                audioTimeRanges = parsedValue
                index += 2
            case "--volatile":
                guard index + 1 < arguments.count,
                      let parsedValue = parseBool(arguments[index + 1])
                else {
                    throw WorkerError.invalidArguments(arguments)
                }
                includeVolatile = parsedValue
                index += 2
            case "--json":
                index += 1
            default:
                throw WorkerError.invalidArguments(arguments)
            }
        }

        guard let inputPath, let locale, let module else {
            throw WorkerError.invalidArguments(arguments)
        }
        return .transcribe(
            TranscriptionRequest(
                inputPath: inputPath,
                locale: locale,
                module: module,
                audioTimeRanges: audioTimeRanges,
                includeVolatile: includeVolatile
            )
        )
    }

    private static func parseBool(_ value: String) -> Bool? {
        switch value {
        case "true":
            true
        case "false":
            false
        default:
            nil
        }
    }
}

public final class AppleSpeechWorker {
    private let runtime: any AppleSpeechRuntime

    public init(runtime: any AppleSpeechRuntime) {
        self.runtime = runtime
    }

    public func run(_ command: WorkerCommand) async throws -> WorkerCommandOutput {
        switch command {
        case .capabilities:
            let capabilities = try await runtime.capabilities()
            return try jsonOutput(capabilities)
        case let .prepare(request):
            let result = try await runtime.prepareAsset(request)
            return try jsonOutput(result)
        case let .transcribe(request):
            let result = try await runtime.transcribe(request)
            return try jsonOutput(result)
        }
    }

    private func jsonOutput<Value: Encodable>(_ value: Value) throws -> WorkerCommandOutput {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(value)
        guard let json = String(data: data, encoding: .utf8) else {
            throw WorkerError.missingFakeResult("json-encoding")
        }
        return WorkerCommandOutput(stdout: json)
    }
}
