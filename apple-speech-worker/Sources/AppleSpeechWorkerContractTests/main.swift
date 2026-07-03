import AppleSpeechWorkerCore
import Foundation

enum ContractTestFailure: Error, CustomStringConvertible {
    case assertion(String)

    var description: String {
        switch self {
        case let .assertion(message):
            return message
        }
    }
}

@main
struct AppleSpeechWorkerContractTests {
    static func main() async {
        do {
            try await capabilitiesCommandWritesOnlyJsonToStdout()
            try parsesCapabilitiesCommand()
            try parsesPrepareCommand()
            try await prepareCommandWritesOnlyJsonToStdout()
            try parsesTranscribeCommand()
            try await transcribeCommandWritesOnlyJsonToStdout()
            try workerErrorDescriptionIsStableForCliStderr()
            try await liveRuntimeReportsSpeechFrameworkCapabilities()
            print("contract-tests: passed")
        } catch {
            FileHandle.standardError.write(Data("contract-tests: failed: \(error)\n".utf8))
            exit(1)
        }
    }

    private static func capabilitiesCommandWritesOnlyJsonToStdout() async throws {
        let runtime = FakeAppleSpeechRuntime()
        runtime.capabilitiesResult = WorkerCapabilities(
            runtime: "apple-speech",
            platform: "macOS",
            osVersion: "26.5",
            supported: true,
            supportedLocales: ["en-US", "zh-CN"],
            modules: WorkerModules(
                speechTranscriber: true,
                dictationTranscriber: true,
                speechDetector: true
            ),
            notes: []
        )
        let worker = AppleSpeechWorker(runtime: runtime)

        let output = try await worker.run(.capabilities)

        guard let json = output.stdout.data(using: .utf8) else {
            throw ContractTestFailure.assertion("stdout is not UTF-8")
        }
        let decoded = try JSONDecoder().decode(WorkerCapabilities.self, from: json)
        try assertEqual(decoded.runtime, "apple-speech", "runtime")
        try assertEqual(decoded.supportedLocales, ["en-US", "zh-CN"], "supportedLocales")
        try assertEqual(output.stderr, "", "stderr")
        try assertFalse(output.stdout.contains("Loading"), "stdout contains Loading")
        try assertFalse(output.stdout.contains("Warning"), "stdout contains Warning")
    }

    private static func liveRuntimeReportsSpeechFrameworkCapabilities() async throws {
        let runtime = LiveAppleSpeechRuntime()

        let capabilities = try await runtime.capabilities()

        try assertEqual(capabilities.runtime, "apple-speech", "runtime")
        try assertEqual(capabilities.platform, "macOS", "platform")
        try assertEqual(capabilities.modules.speechTranscriber, true, "speechTranscriber")
        try assertEqual(capabilities.modules.dictationTranscriber, true, "dictationTranscriber")
        try assertFalse(capabilities.supportedLocales.isEmpty, "supportedLocales is empty")
    }

    private static func parsesCapabilitiesCommand() throws {
        let command = try WorkerCommand.parse(arguments: ["capabilities"])

        try assertEqual(command, .capabilities, "command")
    }

    private static func parsesPrepareCommand() throws {
        let command = try WorkerCommand.parse(
            arguments: [
                "prepare",
                "--locale",
                "zh-CN",
                "--module",
                "speechTranscriber",
                "--json"
            ]
        )

        try assertEqual(
            command,
            .prepare(AssetPreparationRequest(locale: "zh-CN", module: .speechTranscriber)),
            "command"
        )
    }

    private static func prepareCommandWritesOnlyJsonToStdout() async throws {
        let runtime = FakeAppleSpeechRuntime()
        runtime.assetPreparationResult = AssetPreparationResult(
            locale: "zh-CN",
            module: .speechTranscriber,
            supported: true,
            allocated: true,
            downloaded: true,
            durationMs: 1234
        )
        let worker = AppleSpeechWorker(runtime: runtime)

        let output = try await worker.run(
            .prepare(AssetPreparationRequest(locale: "zh-CN", module: .speechTranscriber))
        )

        let json = try requireData(output.stdout)
        let decoded = try JSONDecoder().decode(AssetPreparationResult.self, from: json)
        try assertEqual(decoded.locale, "zh-CN", "locale")
        try assertEqual(decoded.module, .speechTranscriber, "module")
        try assertEqual(decoded.downloaded, true, "downloaded")
        try assertEqual(output.stderr, "", "stderr")
        try assertFalse(output.stdout.contains("Downloading"), "stdout contains Downloading")
    }

    private static func parsesTranscribeCommand() throws {
        let command = try WorkerCommand.parse(
            arguments: [
                "transcribe",
                "--input",
                "/tmp/audio.wav",
                "--locale",
                "en-US",
                "--module",
                "speechTranscriber",
                "--audio-time-ranges",
                "true",
                "--volatile",
                "false",
                "--json"
            ]
        )

        try assertEqual(
            command,
            .transcribe(
                TranscriptionRequest(
                    inputPath: "/tmp/audio.wav",
                    locale: "en-US",
                    module: .speechTranscriber,
                    audioTimeRanges: true,
                    includeVolatile: false
                )
            ),
            "command"
        )
    }

    private static func transcribeCommandWritesOnlyJsonToStdout() async throws {
        let runtime = FakeAppleSpeechRuntime()
        runtime.transcriptionResult = TranscriptionResult(
            jobId: nil,
            engine: "apple-speech",
            module: .speechTranscriber,
            locale: "en-US",
            text: "hello world",
            segments: [
                TranscriptionSegment(
                    id: 0,
                    start: 0.0,
                    end: 1.0,
                    text: "hello world",
                    isFinal: true,
                    confidence: nil,
                    speaker: nil
                )
            ],
            metadata: TranscriptionMetadata(
                local: true,
                appleApi: true,
                volatileIncluded: false,
                timingGranularity: "segment",
                assetManagedBySystem: true,
                durationMs: 42
            )
        )
        let worker = AppleSpeechWorker(runtime: runtime)

        let output = try await worker.run(
            .transcribe(
                TranscriptionRequest(
                    inputPath: "/tmp/audio.wav",
                    locale: "en-US",
                    module: .speechTranscriber,
                    audioTimeRanges: true,
                    includeVolatile: false
                )
            )
        )

        let json = try requireData(output.stdout)
        let decoded = try JSONDecoder().decode(TranscriptionResult.self, from: json)
        try assertEqual(decoded.engine, "apple-speech", "engine")
        try assertEqual(decoded.text, "hello world", "text")
        try assertEqual(decoded.segments.count, 1, "segments.count")
        try assertEqual(decoded.metadata.timingGranularity, "segment", "timingGranularity")
        try assertEqual(output.stderr, "", "stderr")
        try assertContains(output.stdout, "\"jobId\":null", "stdout contains jobId null")
        try assertContains(output.stdout, "\"confidence\":null", "stdout contains confidence null")
        try assertContains(output.stdout, "\"speaker\":null", "stdout contains speaker null")
        try assertFalse(output.stdout.contains("Transcribing"), "stdout contains Transcribing")
    }

    private static func workerErrorDescriptionIsStableForCliStderr() throws {
        let error = WorkerError.invalidArguments(["prepare", "--locale"])

        try assertEqual(
            error.description,
            "invalid arguments: prepare --locale",
            "WorkerError.invalidArguments description"
        )
    }

    private static func assertEqual<Value: Equatable>(
        _ actual: Value,
        _ expected: Value,
        _ label: String
    ) throws {
        guard actual == expected else {
            throw ContractTestFailure.assertion("\(label): expected \(expected), got \(actual)")
        }
    }

    private static func assertFalse(_ value: Bool, _ label: String) throws {
        guard !value else {
            throw ContractTestFailure.assertion(label)
        }
    }

    private static func assertContains(
        _ value: String,
        _ expectedSubstring: String,
        _ label: String
    ) throws {
        guard value.contains(expectedSubstring) else {
            throw ContractTestFailure.assertion(label)
        }
    }

    private static func requireData(_ value: String) throws -> Data {
        guard let data = value.data(using: .utf8) else {
            throw ContractTestFailure.assertion("value is not UTF-8")
        }
        return data
    }
}
