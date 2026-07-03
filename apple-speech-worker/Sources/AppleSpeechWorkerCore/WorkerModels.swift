import Foundation

public struct WorkerModules: Codable, Equatable, Sendable {
    public let speechTranscriber: Bool
    public let dictationTranscriber: Bool
    public let speechDetector: Bool

    public init(
        speechTranscriber: Bool,
        dictationTranscriber: Bool,
        speechDetector: Bool
    ) {
        self.speechTranscriber = speechTranscriber
        self.dictationTranscriber = dictationTranscriber
        self.speechDetector = speechDetector
    }
}

public struct WorkerCapabilities: Codable, Equatable, Sendable {
    public let runtime: String
    public let platform: String
    public let osVersion: String
    public let supported: Bool
    public let supportedLocales: [String]
    public let modules: WorkerModules
    public let notes: [String]

    public init(
        runtime: String,
        platform: String,
        osVersion: String,
        supported: Bool,
        supportedLocales: [String],
        modules: WorkerModules,
        notes: [String]
    ) {
        self.runtime = runtime
        self.platform = platform
        self.osVersion = osVersion
        self.supported = supported
        self.supportedLocales = supportedLocales
        self.modules = modules
        self.notes = notes
    }
}

public struct WorkerCommandOutput: Equatable, Sendable {
    public let stdout: String
    public let stderr: String

    public init(stdout: String, stderr: String = "") {
        self.stdout = stdout
        self.stderr = stderr
    }
}

public enum WorkerSpeechModule: String, Codable, Equatable, Sendable {
    case speechTranscriber
    case dictationTranscriber
}

public struct AssetPreparationRequest: Equatable, Sendable {
    public let locale: String
    public let module: WorkerSpeechModule

    public init(locale: String, module: WorkerSpeechModule) {
        self.locale = locale
        self.module = module
    }
}

public struct AssetPreparationResult: Codable, Equatable, Sendable {
    public let locale: String
    public let module: WorkerSpeechModule
    public let supported: Bool
    public let allocated: Bool
    public let downloaded: Bool
    public let durationMs: Int

    public init(
        locale: String,
        module: WorkerSpeechModule,
        supported: Bool,
        allocated: Bool,
        downloaded: Bool,
        durationMs: Int
    ) {
        self.locale = locale
        self.module = module
        self.supported = supported
        self.allocated = allocated
        self.downloaded = downloaded
        self.durationMs = durationMs
    }
}

public struct TranscriptionRequest: Equatable, Sendable {
    public let inputPath: String
    public let locale: String
    public let module: WorkerSpeechModule
    public let audioTimeRanges: Bool
    public let includeVolatile: Bool

    public init(
        inputPath: String,
        locale: String,
        module: WorkerSpeechModule,
        audioTimeRanges: Bool,
        includeVolatile: Bool
    ) {
        self.inputPath = inputPath
        self.locale = locale
        self.module = module
        self.audioTimeRanges = audioTimeRanges
        self.includeVolatile = includeVolatile
    }
}

public struct TranscriptionSegment: Codable, Equatable, Sendable {
    public let id: Int
    public let start: Double
    public let end: Double
    public let text: String
    public let isFinal: Bool
    public let confidence: Double?
    public let speaker: String?

    public init(
        id: Int,
        start: Double,
        end: Double,
        text: String,
        isFinal: Bool,
        confidence: Double?,
        speaker: String?
    ) {
        self.id = id
        self.start = start
        self.end = end
        self.text = text
        self.isFinal = isFinal
        self.confidence = confidence
        self.speaker = speaker
    }

    enum CodingKeys: String, CodingKey {
        case id
        case start
        case end
        case text
        case isFinal
        case confidence
        case speaker
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(start, forKey: .start)
        try container.encode(end, forKey: .end)
        try container.encode(text, forKey: .text)
        try container.encode(isFinal, forKey: .isFinal)
        try container.encode(confidence, forKey: .confidence)
        try container.encode(speaker, forKey: .speaker)
    }
}

public struct TranscriptionMetadata: Codable, Equatable, Sendable {
    public let local: Bool
    public let appleApi: Bool
    public let volatileIncluded: Bool
    public let timingGranularity: String
    public let assetManagedBySystem: Bool
    public let durationMs: Int

    public init(
        local: Bool,
        appleApi: Bool,
        volatileIncluded: Bool,
        timingGranularity: String,
        assetManagedBySystem: Bool,
        durationMs: Int
    ) {
        self.local = local
        self.appleApi = appleApi
        self.volatileIncluded = volatileIncluded
        self.timingGranularity = timingGranularity
        self.assetManagedBySystem = assetManagedBySystem
        self.durationMs = durationMs
    }
}

public struct TranscriptionResult: Codable, Equatable, Sendable {
    public let jobId: String?
    public let engine: String
    public let module: WorkerSpeechModule
    public let locale: String
    public let text: String
    public let segments: [TranscriptionSegment]
    public let metadata: TranscriptionMetadata

    public init(
        jobId: String?,
        engine: String,
        module: WorkerSpeechModule,
        locale: String,
        text: String,
        segments: [TranscriptionSegment],
        metadata: TranscriptionMetadata
    ) {
        self.jobId = jobId
        self.engine = engine
        self.module = module
        self.locale = locale
        self.text = text
        self.segments = segments
        self.metadata = metadata
    }

    enum CodingKeys: String, CodingKey {
        case jobId
        case engine
        case module
        case locale
        case text
        case segments
        case metadata
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(jobId, forKey: .jobId)
        try container.encode(engine, forKey: .engine)
        try container.encode(module, forKey: .module)
        try container.encode(locale, forKey: .locale)
        try container.encode(text, forKey: .text)
        try container.encode(segments, forKey: .segments)
        try container.encode(metadata, forKey: .metadata)
    }
}
