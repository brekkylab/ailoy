mod python;

use proc_macro::TokenStream;
use python::*;
use quote::quote;
use syn::{FnArg, Item, ItemFn, Pat, Type, parse_macro_input, parse_quote};

#[proc_macro_attribute]
pub fn maybe_send_sync(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as Item);

    match input {
        Item::Type(item) => {
            let vis = &item.vis;
            let ident = &item.ident;
            let generics = &item.generics;
            let ty = &item.ty;

            // Create a version with Send + Sync appended
            let ty_with_bounds = if let Type::TraitObject(trait_obj) = &**ty {
                let mut bounds = trait_obj.bounds.clone();
                bounds.push(parse_quote!(Send));
                bounds.push(parse_quote!(Sync));

                let mut new_trait_obj = trait_obj.clone();
                new_trait_obj.bounds = bounds;

                Type::TraitObject(new_trait_obj)
            } else {
                parse_quote!(#ty + Send + Sync)
            };

            let output = quote! {
                #[cfg(not(target_arch = "wasm32"))]
                #vis type #ident #generics = #ty_with_bounds;

                #[cfg(target_arch = "wasm32")]
                #vis type #ident #generics = #ty;
            };

            output.into()
        }
        Item::Trait(item) => {
            let attrs = &item.attrs;
            let vis = &item.vis;
            let unsafety = &item.unsafety;
            let trait_token = &item.trait_token;
            let ident = &item.ident;
            let generics = &item.generics;
            let items = &item.items;

            // Handle supertraits
            let supertraits = &item.supertraits;
            let has_supertraits = !supertraits.is_empty();

            // Create version with Send + Sync
            let mut supertraits_with_bounds = supertraits.clone();
            supertraits_with_bounds.push(parse_quote!(Send));
            supertraits_with_bounds.push(parse_quote!(Sync));

            let output = if has_supertraits {
                quote! {
                    #[cfg(not(target_arch = "wasm32"))]
                    #(#attrs)*
                    #vis #unsafety #trait_token #ident #generics: #supertraits_with_bounds {
                        #(#items)*
                    }

                    #[cfg(target_arch = "wasm32")]
                    #(#attrs)*
                    #vis #unsafety #trait_token #ident #generics: #supertraits {
                        #(#items)*
                    }
                }
            } else {
                quote! {
                    #[cfg(not(target_arch = "wasm32"))]
                    #(#attrs)*
                    #vis #unsafety #trait_token #ident #generics: Send + Sync {
                        #(#items)*
                    }

                    #[cfg(target_arch = "wasm32")]
                    #(#attrs)*
                    #vis #unsafety #trait_token #ident #generics {
                        #(#items)*
                    }
                }
            };

            output.into()
        }
        _ => {
            return syn::Error::new_spanned(
                quote! { #input },
                "maybe_send_sync can only be applied to type aliases or trait definitions",
            )
            .to_compile_error()
            .into();
        }
    }
}

#[proc_macro_attribute]
pub fn multi_platform_test(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let fn_vis = &input_fn.vis;
    let fn_inputs = &input_fn.sig.inputs;
    let fn_output = &input_fn.sig.output;
    let fn_body = &input_fn.block;
    let fn_attrs = &input_fn.attrs;

    let output = quote! {
        #(#fn_attrs)*
        #[cfg(not(target_arch = "wasm32"))]
        #[tokio::test]
        #fn_vis async fn #fn_name(#fn_inputs) #fn_output {
            #[cfg(not(target_arch = "wasm32"))]
            let _ = env_logger::builder().is_test(true).try_init();
            #fn_body
        }

        #(#fn_attrs)*
        #[cfg(target_arch = "wasm32")]
        #[wasm_bindgen_test::wasm_bindgen_test]
        #fn_vis async fn #fn_name(#fn_inputs) #fn_output {
            #fn_body
        }
    };

    output.into()
}

#[proc_macro_attribute]
pub fn multi_platform_async_trait(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_item = parse_macro_input!(item as Item);

    // Use literal token strings instead of path references
    let output = quote! {
        #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
        #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
        #input_item
    };

    output.into()
}

#[proc_macro_derive(PyStringEnum)]
pub fn derive_py_string_enum(input: TokenStream) -> TokenStream {
    py_string_enum(input)
}

/// Attribute macro for generating a tool factory function from an async function.
///
/// Usage:
/// ```ignore
/// #[tool(description = "Get current temperature for a city")]
/// async fn get_temperature(location: String, unit: String) -> anyhow::Result<ailoy::value::Value> {
///     Ok(ailoy::to_value!("40"))
/// }
/// ```
///
/// This keeps the original function and generates a `get_temperature_tool() -> Tool` factory.
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);

    // Parse description from attr: description = "..."
    let attr_str = attr.to_string();
    let description = parse_tool_description(&attr_str);

    let fn_name = &input_fn.sig.ident;
    let tool_fn_name = quote::format_ident!("{}_tool", fn_name);
    let fn_name_str = fn_name.to_string();

    // Collect parameter names and types from the function signature
    let mut param_names: Vec<syn::Ident> = Vec::new();
    let mut param_types: Vec<Box<Type>> = Vec::new();

    for arg in &input_fn.sig.inputs {
        if let FnArg::Typed(pat_type) = arg {
            if let Pat::Ident(pat_ident) = &*pat_type.pat {
                param_names.push(pat_ident.ident.clone());
                param_types.push(pat_type.ty.clone());
            }
        }
    }

    let param_name_strs: Vec<String> = param_names.iter().map(|n| n.to_string()).collect();

    // Build the schema property entries for each parameter
    let schema_entries = param_name_strs.iter().zip(param_types.iter()).map(|(name, ty)| {
        quote! {
            #name: serde_json::to_value(schema_gen.subschema_for::<#ty>()).unwrap()
        }
    });

    // Build required array
    let required_entries = param_name_strs.iter().map(|name| {
        quote! { #name }
    });

    // Build handler argument extraction
    let handler_extractions = param_names.iter().zip(param_name_strs.iter()).zip(param_types.iter()).map(|((ident, name), ty)| {
        quote! {
            let #ident: #ty = serde_json::from_value(
                serde_json::to_value(
                    args.as_object()
                        .ok_or_else(|| anyhow::anyhow!("Tool args must be a JSON object"))?
                        .get(#name)
                        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: {}", #name))?
                ).map_err(anyhow::Error::from)?
            )?;
        }
    });

    let call_args = param_names.iter().map(|n| quote! { #n });

    let desc_token = match &description {
        Some(d) => quote! { .description(#d) },
        None => quote! {},
    };

    let output = quote! {
        #input_fn

        pub fn #tool_fn_name() -> ailoy::tool::Tool {
            ailoy::tool::Tool::builder()
                .name(#fn_name_str)
                #desc_token
                .parameters({
                    let mut schema_gen = schemars::r#gen::SchemaGenerator::default();
                    let properties = serde_json::json!({
                        #(#schema_entries),*
                    });
                    serde_json::from_value::<ailoy::value::Value>(serde_json::json!({
                        "type": "object",
                        "properties": properties,
                        "required": [#(#required_entries),*]
                    })).unwrap()
                })
                .handler(|args: ailoy::value::Value| async move {
                    #(#handler_extractions)*
                    #fn_name(#(#call_args),*).await
                })
                .build()
                .unwrap()
        }
    };

    output.into()
}

/// Parse the description value from the tool attribute string.
/// Expected format: `description = "some text"`
fn parse_tool_description(attr_str: &str) -> Option<String> {
    let trimmed = attr_str.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Find the pattern: description = "..."
    if let Some(idx) = trimmed.find("description") {
        let rest = &trimmed[idx + "description".len()..].trim_start();
        if let Some(rest) = rest.strip_prefix('=') {
            let rest = rest.trim_start();
            // Strip surrounding quotes
            if rest.starts_with('"') && rest.ends_with('"') && rest.len() >= 2 {
                return Some(rest[1..rest.len() - 1].to_string());
            }
        }
    }
    None
}
