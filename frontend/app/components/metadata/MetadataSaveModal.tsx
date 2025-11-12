import { Show } from "solid-js";
import type { JSX } from "solid-js";
import { supportsFilePicker } from "../shared/utils";

interface MetadataSaveModalProps {
  open: boolean;
  saving: boolean;
  defaultPath: string;
  localFileName: string;
  onClose: () => void;
  onSubmit: JSX.EventHandlerUnion<HTMLFormElement, SubmitEvent>;
  onChooseLocalPath: () => void;
  formRef?: (el: HTMLFormElement) => void;
}

export function MetadataSaveModal(props: MetadataSaveModalProps) {
  return (
    <Show when={props.open}>
      <div
        class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4"
        role="dialog"
        aria-modal="true"
      >
        <div class="w-full max-w-lg rounded-lg bg-white shadow-xl">
          <header class="flex items-center justify-between border-b px-6 py-4">
            <h2 class="text-lg font-semibold text-gray-900">Save Metadata</h2>
            <button
              type="button"
              class="text-gray-500 hover:text-gray-800"
              aria-label="Close"
              onClick={props.onClose}
            >
              ✕
            </button>
          </header>
          <form ref={props.formRef} onSubmit={props.onSubmit} class="space-y-4 px-6 py-6">
            <p class="text-sm text-gray-600">
              Save the current metadata result set to a backend-visible path (default:{" "}
              <code class="rounded bg-gray-100 px-1 py-0.5 text-xs text-gray-800">{props.defaultPath}</code>) or pick a local
              destination on your machine.
            </p>
            <label class="block text-sm font-medium text-gray-700" for="save_path">
              Destination path
              <input
                type="text"
                id="save_path"
                name="save_path"
                defaultValue={props.defaultPath}
                placeholder={props.defaultPath}
                class="mt-1 w-full rounded-md border border-gray-300 p-2"
              />
            </label>
            <div class="text-xs text-gray-500">
              We attempt to persist the file server-side; if that fails, a download starts in your browser.
            </div>

            <div class="rounded-md border border-dashed border-gray-300 p-3">
              <div class="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p class="text-sm font-medium text-gray-700">Local save location (optional)</p>
                  <p class="text-xs text-gray-500">
                    {supportsFilePicker()
                      ? props.localFileName
                        ? `Selected: ${props.localFileName}`
                        : "No local file chosen yet."
                      : "Your browser will prompt for a download instead."}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={props.onChooseLocalPath}
                  disabled={!supportsFilePicker()}
                  class="rounded-md border border-gray-300 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed disabled:text-gray-400"
                >
                  Choose location
                </button>
              </div>
            </div>

            <div class="flex items-center justify-end gap-2">
              <button
                type="button"
                class="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
                onClick={props.onClose}
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={props.saving}
                class="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-blue-300"
              >
                {props.saving ? "Saving..." : "Save"}
              </button>
            </div>
          </form>
        </div>
      </div>
    </Show>
  );
}

