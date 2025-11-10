import { createHandler, StartServer } from "@solidjs/start/server";
import Document from "./document";

export default createHandler((event) => <StartServer event={event} document={Document} />);
