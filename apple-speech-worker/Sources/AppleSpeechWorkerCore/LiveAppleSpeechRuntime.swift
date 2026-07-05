import Foundation
import AVFAudio
import CoreMedia
import Speech

public final class LiveAppleSpeechRuntime: AppleSpeechRuntime, @unchecked Sendable {
    public init() {}

    public func capabilities() async throws -> WorkerCapabilities {
        let speechLocales = await SpeechTranscriber.supportedLocales
        let dictationLocales = await DictationTranscriber.supportedLocales
        let supportedLocaleIdentifiers = Set(
            speechLocales.map(\.identifier) + dictationLocales.map(\.identifier)
        ).sorted()
        let notes = capabilityNotes(
            speechTranscriberAvailable: SpeechTranscriber.isAvailable,
            supportedLocales: supportedLocaleIdentifiers
        )

        return WorkerCapabilities(
            runtime: "apple-speech",
            platform: "macOS",
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            supported: SpeechTranscriber.isAvailable && !supportedLocaleIdentifiers.isEmpty,
            supportedLocales: supportedLocaleIdentifiers,
            modules: WorkerModules(
                speechTranscriber: SpeechTranscriber.isAvailable,
                dictationTranscriber: !dictationLocales.isEmpty,
                speechDetector: true
            ),
            notes: notes
        )
    }

    public func prepareAsset(_ request: AssetPreparationRequest) async throws -> AssetPreparationResult {
        let startedAt = Date()
        let locale = Locale(identifier: request.locale)
        let module = speechModule(for: request.module, locale: locale)
        let initialStatus = await AssetInventory.status(forModules: [module])

        guard initialStatus != .unsupported else {
            return AssetPreparationResult(
                locale: request.locale,
                module: request.module,
                supported: false,
                allocated: false,
                downloaded: false,
                durationMs: durationMs(since: startedAt)
            )
        }

        let allocated = try await AssetInventory.reserve(locale: locale)
        let finalStatus = await AssetInventory.status(forModules: [module])
        let downloaded = finalStatus == .installed

        return AssetPreparationResult(
            locale: request.locale,
            module: request.module,
            supported: finalStatus != .unsupported,
            allocated: allocated || downloaded,
            downloaded: downloaded,
            durationMs: durationMs(since: startedAt)
        )
    }

    public func transcribe(_ request: TranscriptionRequest) async throws -> TranscriptionResult {
        guard request.module == .speechTranscriber else {
            throw WorkerError.unsupportedModule(request.module)
        }

        let startedAt = Date()
        _ = try await prepareAsset(
            AssetPreparationRequest(locale: request.locale, module: request.module)
        )

        let locale = Locale(identifier: request.locale)
        let transcriber = SpeechTranscriber(
            locale: locale,
            transcriptionOptions: [],
            reportingOptions: reportingOptions(includeVolatile: request.includeVolatile),
            attributeOptions: attributeOptions(includeAudioTimeRanges: request.audioTimeRanges)
        )
        let audioFile = try AVAudioFile(forReading: URL(fileURLWithPath: request.inputPath))
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let resultsTask = Task {
            var collectedResults: [SpeechTranscriber.Result] = []
            for try await result in transcriber.results {
                collectedResults.append(result)
            }
            return collectedResults
        }
        defer {
            resultsTask.cancel()
        }

        _ = try await analyzer.analyzeSequence(from: audioFile)
        try await analyzer.finalizeAndFinishThroughEndOfInput()

        let speechResults = try await resultsTask.value
        let segments = speechResults.enumerated().map { index, result in
            transcriptionSegment(id: index, result: result)
        }
        let text = segments.map(\.text).joined(separator: " ")

        return TranscriptionResult(
            jobId: nil,
            engine: "apple-speech",
            module: request.module,
            locale: request.locale,
            text: text,
            segments: segments,
            metadata: TranscriptionMetadata(
                local: true,
                appleApi: true,
                volatileIncluded: request.includeVolatile,
                timingGranularity: request.audioTimeRanges ? "segment" : "unknown",
                assetManagedBySystem: true,
                durationMs: durationMs(since: startedAt)
            )
        )
    }

    private func capabilityNotes(
        speechTranscriberAvailable: Bool,
        supportedLocales: [String]
    ) -> [String] {
        var notes: [String] = []
        if !speechTranscriberAvailable {
            notes.append("SpeechTranscriber.isAvailable is false.")
        }
        if supportedLocales.isEmpty {
            notes.append("No Speech framework locales were reported.")
        }
        return notes
    }

    private func speechModule(
        for module: WorkerSpeechModule,
        locale: Locale
    ) -> any SpeechModule {
        switch module {
        case .speechTranscriber:
            SpeechTranscriber(
                locale: locale,
                preset: .timeIndexedTranscriptionWithAlternatives
            )
        case .dictationTranscriber:
            DictationTranscriber(
                locale: locale,
                preset: .timeIndexedLongDictation
            )
        }
    }

    private func durationMs(since startedAt: Date) -> Int {
        Int(Date().timeIntervalSince(startedAt) * 1000)
    }

    private func reportingOptions(
        includeVolatile: Bool
    ) -> Set<SpeechTranscriber.ReportingOption> {
        includeVolatile ? [.volatileResults] : []
    }

    private func attributeOptions(
        includeAudioTimeRanges: Bool
    ) -> Set<SpeechTranscriber.ResultAttributeOption> {
        var options: Set<SpeechTranscriber.ResultAttributeOption> = [.transcriptionConfidence]
        if includeAudioTimeRanges {
            options.insert(.audioTimeRange)
        }
        return options
    }

    private func transcriptionSegment(
        id: Int,
        result: SpeechTranscriber.Result
    ) -> TranscriptionSegment {
        let timeRange = speechResultTimeRange(result)
        let start = timeRange.map { seconds($0.start) } ?? 0.0
        let duration = timeRange.map { seconds($0.duration) } ?? 0.0
        return TranscriptionSegment(
            id: id,
            start: start,
            end: start + duration,
            text: String(result.text.characters),
            isFinal: true,
            confidence: nil,
            speaker: nil
        )
    }

    private func speechResultTimeRange(_ result: SpeechTranscriber.Result) -> CMTimeRange? {
        // Avoid binding to beta SDK variations of SpeechTranscriber.Result.range.
        Mirror(reflecting: result)
            .children
            .first { $0.label == "range" }?
            .value as? CMTimeRange
    }

    private func seconds(_ time: CMTime) -> Double {
        let value = CMTimeGetSeconds(time)
        return value.isFinite ? value : 0.0
    }
}
