"""RDM's tensor names (BFL's original layout: q/k/v fused, 'model.' prefix) mapped onto diffusers' (klein and the painter)."""
D, MLP = 3072, 9216


def pieces(k):
    """RDM key -> [(diffusers key, rows slice or None)]"""
    k = k.removeprefix('model.')
    parts = k.split('.')
    if parts[0] == 'double_blocks':
        b, rest = parts[1], '.'.join(parts[2:])
        P = f'transformer_blocks.{b}.'
        table = {
            'img_attn.qkv.weight': [(P + 'attn.to_q.weight', (0, D)), (P + 'attn.to_k.weight', (D, 2 * D)),
                                    (P + 'attn.to_v.weight', (2 * D, 3 * D))],
            'txt_attn.qkv.weight': [(P + 'attn.add_q_proj.weight', (0, D)), (P + 'attn.add_k_proj.weight', (D, 2 * D)),
                                    (P + 'attn.add_v_proj.weight', (2 * D, 3 * D))],
            'img_attn.proj.weight': [(P + 'attn.to_out.0.weight', None)],
            'txt_attn.proj.weight': [(P + 'attn.to_add_out.weight', None)],
            'img_mlp.0.weight': [(P + 'ff.linear_in.weight', None)],
            'img_mlp.2.weight': [(P + 'ff.linear_out.weight', None)],
            'txt_mlp.0.weight': [(P + 'ff_context.linear_in.weight', None)],
            'txt_mlp.2.weight': [(P + 'ff_context.linear_out.weight', None)],
            'img_attn.norm.query_norm.scale': [(P + 'attn.norm_q.weight', None)],
            'img_attn.norm.key_norm.scale': [(P + 'attn.norm_k.weight', None)],
            'txt_attn.norm.query_norm.scale': [(P + 'attn.norm_added_q.weight', None)],
            'txt_attn.norm.key_norm.scale': [(P + 'attn.norm_added_k.weight', None)],
        }
        return table[rest]
    if parts[0] == 'single_blocks':
        b, rest = parts[1], '.'.join(parts[2:])
        P = f'single_transformer_blocks.{b}.'
        return [{'linear1.weight': (P + 'attn.to_qkv_mlp_proj.weight', None),
                 'linear2.weight': (P + 'attn.to_out.weight', None),
                 'norm.query_norm.scale': (P + 'attn.norm_q.weight', None),
                 'norm.key_norm.scale': (P + 'attn.norm_k.weight', None)}[rest]]
    return [{
        'double_stream_modulation_img.lin.weight': ('double_stream_modulation_img.linear.weight', None),
        'double_stream_modulation_txt.lin.weight': ('double_stream_modulation_txt.linear.weight', None),
        'single_stream_modulation.lin.weight': ('single_stream_modulation.linear.weight', None),
        'final_layer.adaLN_modulation.1.weight': ('norm_out.linear.weight', 'swap'),  # diffusers: scale | shift
        'final_layer.linear.weight': ('proj_out.weight', None),
        'img_in.weight': ('x_embedder.weight', None),
        'txt_in.weight': ('context_embedder.weight', None),
        'time_in.in_layer.weight': ('time_guidance_embed.timestep_embedder.linear_1.weight', None),
        'time_in.out_layer.weight': ('time_guidance_embed.timestep_embedder.linear_2.weight', None),
    }[k]]
