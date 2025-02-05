import "@typespec/http";
import "@typespec/openapi";

import "../common/models.tsp";
import "../common/errors.tsp";
import "./models.tsp";

using TypeSpec.Http;
using TypeSpec.OpenAPI;

namespace OpenAI;

@route("/assistants")
interface Assistants {
  @post
  @operationId("createAssistant")
  @tag("Assistants")
  @summary("Create an assistant with a model and instructions.")
  createAssistant(
    @body assistant: CreateAssistantRequest,
  ): AssistantObject | ErrorResponse;

  @get
  @operationId("listAssistants")
  @tag("Assistants")
  @summary("Returns a list of assistants.")
  listAssistants(
    /** 
     * A limit on the number of objects to be returned. Limit can range between 1 and 100, and the
     * default is 20.
     */
    @query limit?: int32 = 20;

    /**
     * Sort order by the `created_at` timestamp of the objects. `asc` for ascending order and`desc`
     * for descending order.
     */
    @query order?: ListOrder = ListOrder.desc;

    /**
     * A cursor for use in pagination. `after` is an object ID that defines your place in the list.
     * For instance, if you make a list request and receive 100 objects, ending with obj_foo, your
     * subsequent call can include after=obj_foo in order to fetch the next page of the list.
     */
    @query after?: string;

    /**
     * A cursor for use in pagination. `before` is an object ID that defines your place in the list.
     * For instance, if you make a list request and receive 100 objects, ending with obj_foo, your
     * subsequent call can include before=obj_foo in order to fetch the previous page of the list.
     */
    @query before?: string;
  ): ListAssistantsResponse | ErrorResponse;

  @route("{assistant_id}")
  @get
  @operationId("getAssistant")
  @tag("Assistants")
  @summary("Retrieves an assistant.")
  getAssistant(
    /** The ID of the assistant to retrieve. */
    @path assistant_id: string,
  ): AssistantObject | ErrorResponse;

  @route("{assistant_id}")
  @post
  @operationId("modifyAssistant")
  @tag("Assistants")
  @summary("Modifies an assistant.")
  modifyAssistant(
    /** The ID of the assistant to modify. */
    @path assistant_id: string,

    @body assistant: ModifyAssistantRequest,
  ): AssistantObject | ErrorResponse;

  @route("{assistant_id}")
  @delete
  @operationId("deleteAssistant")
  @tag("Assistants")
  @summary("Delete an assistant.")
  deleteAssistant(
    /** The ID of the assistant to delete. */
    @path assistant_id: string,
  ): DeleteAssistantResponse | ErrorResponse;

  @route("{assistant_id}/files")
  @post
  @operationId("createAssistantFile")
  @tag("Assistants")
  @summary("""
  Create an assistant file by attaching a [File](/docs/api-reference/files) to a
  [assistant](/docs/api-reference/assistants).
  """)
  createAssistantFile(
    /** The ID of the assistant for which to create a file. */
    @path assistant_id: string,
    @body file: CreateAssistantFileRequest,
  ): AssistantFileObject | ErrorResponse;

  @route("{assistant_id}/files")
  @get
  @operationId("listAssistantFiles")
  @tag("Assistants")
  @summary("Returns a list of assistant files.")
  listAssistantFiles(
    /** The ID of the assistant the file belongs to. */
    @path assistant_id: string,

    /** 
     * A limit on the number of objects to be returned. Limit can range between 1 and 100, and the
     * default is 20.
     */
    @query limit?: int32 = 20;

    /**
     * Sort order by the `created_at` timestamp of the objects. `asc` for ascending order and`desc`
     * for descending order.
     */
    @query order?: ListOrder = ListOrder.desc;

    /**
     * A cursor for use in pagination. `after` is an object ID that defines your place in the list.
     * For instance, if you make a list request and receive 100 objects, ending with obj_foo, your
     * subsequent call can include after=obj_foo in order to fetch the next page of the list.
     */
    @query after?: string;

    /**
     * A cursor for use in pagination. `before` is an object ID that defines your place in the list.
     * For instance, if you make a list request and receive 100 objects, ending with obj_foo, your
     * subsequent call can include before=obj_foo in order to fetch the previous page of the list.
     */
    @query before?: string;
  ): ListAssistantFilesResponse | ErrorResponse;

  @route("{assistant_id}/files/{file_id}")
  @get
  @operationId("getAssistantFile")
  @tag("Assistants")
  @summary("Retrieves an assistant file.")
  getAssistantFile(
    /** The ID of the assistant the file belongs to. */
    @path assistant_id: string,

    /** The ID of the file we're getting. */
    @path file_id: string,
  ): AssistantFileObject | ErrorResponse;

  @route("{assistant_id}/files/{file_id}")
  @delete
  @operationId("deleteAssistantFile")
  @tag("Assistants")
  @summary("Delete an assistant file.")
  deleteAssistantFile(
    /** The ID of the assistant the file belongs to. */
    @path assistant_id: string,
  
    /** The ID of the file to delete. */
    @path file_id: string,
  ): DeleteAssistantFileResponse | ErrorResponse;
}
