import AppleSpeechWorkerCore
import Foundation

@main
struct AppleSpeechWorkerCLI {
    static func main() async {
        do {
            let command = try WorkerCommand.parse(arguments: Array(CommandLine.arguments.dropFirst()))
            let worker = AppleSpeechWorker(runtime: LiveAppleSpeechRuntime())
            let output = try await worker.run(command)
            write(output.stdout + "\n", to: .standardOutput)
            if !output.stderr.isEmpty {
                write(output.stderr, to: .standardError)
            }
        } catch {
            write("apple-speech-worker: \(error)\n", to: .standardError)
            exit(1)
        }
    }

    private static func write(_ value: String, to handle: FileHandle) {
        handle.write(Data(value.utf8))
    }
}
