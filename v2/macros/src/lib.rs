use proc_macro::TokenStream;
use quote::quote;
use syn::{Item, ItemFn, Type, parse_macro_input, parse_quote};

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
