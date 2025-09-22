use proc_macro::TokenStream;
use quote::quote;
use syn::{Item, ItemFn, parse_macro_input};

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
