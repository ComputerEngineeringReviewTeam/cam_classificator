```
{
 "form": {
   "attributes": {
     // HTML attributes for the <form> element
     // use literal URL in "action" - not Jinja functions
   }
 },
  "fields": [
    {
      // fields
      // for each one a single <input> element will be created
      "field_name": // will be used by HTML <input> and by database, MUST be unique 
      "datatype":   // one from ["str", "bool", "int", "float"]
      "default":    // default value
      "attributes": {
        // HTML attributes for the <input> element 
        // "type" is required!
      }

    }
  ]
}
```