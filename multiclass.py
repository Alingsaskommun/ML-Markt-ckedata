import os
from os.path import join, basename

from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.analyzer import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
from rastervision.pytorch_backend.examples.utils import (get_scene_info,
                                                         save_image_crop)
from rastervision.pytorch_backend.examples.semantic_segmentation.utils import (
    example_multiband_transform, example_rgb_transform, imagenet_stats,
    Unnormalize)

 # Lägg in träningsområdena nedan
 # Fyll på med "CLASS_NAMES" och välj färg med hexakod på den nya klassen (färgen visas sedan i rgb output i predict mappen)

TRAIN_IDS = [
    '1_1', '2_1'
]
VAL_IDS = ['1_1', '2_1']

CLASS_NAMES = [
    'road', 'building', 'canopy'
]
CLASS_COLORS = [
    '#ffff00', '#0000ff', '#00ffff'
]

 # Om test: bool = True så körs koden enligt test mode definierat längre ner med färre epoker och mindre batch_sz
def get_config(runner,
               multiband: bool = False,
               external_model: bool = True,
               augment: bool = False,
               nochip: bool = True,
               test: bool = False):
    root_uri = '/opt/data/output'
    raw_uri = '/opt/data/data_input'
    processed_uri = '/opt/data/data_input/crops'
   
    train_ids = TRAIN_IDS
    val_ids = VAL_IDS

    if test:
        train_ids = train_ids[:2]
        val_ids = val_ids[:2]

    if multiband:
        # använd alla 4 kanaler
        channel_order = [0, 1, 2, 3]
        channel_display_groups = {'RGB': (0, 1, 2), 'EL': (3, )}
        aug_transform = example_multiband_transform
    else:
        # använd elevation, red, & green kanaler
        channel_order = [3, 0, 1]
        channel_display_groups = None
        aug_transform = example_rgb_transform

    if augment:
        mu, std = imagenet_stats['mean'], imagenet_stats['std']
        mu, std = mu[channel_order], std[channel_order]

        base_transform = A.Normalize(mean=mu.tolist(), std=std.tolist())
        plot_transform = Unnormalize(mean=mu, std=std)

        aug_transform = A.to_dict(aug_transform)
        base_transform = A.to_dict(base_transform)
        plot_transform = A.to_dict(plot_transform)
    else:
        aug_transform = None
        base_transform = None
        plot_transform = None

    class_config = ClassConfig(names=CLASS_NAMES, colors=CLASS_COLORS)
    class_config.ensure_null_class()

    def make_scene(id) -> SceneConfig:
        id = id.replace('-', '_')
        raster_uri = f'{raw_uri}/images/sample_{id}_RGBEL.tif'
        label_uri = f'{raw_uri}/labels/sample_{id}_label.geojson'

        if test:
            crop_uri = join(processed_uri, 'crops', basename(raster_uri))
            label_crop_uri = join(processed_uri, 'crops', basename(label_uri))
            save_image_crop(
                raster_uri,
                crop_uri,
                label_uri=label_uri,
                label_crop_uri=label_crop_uri,
                size=600,
                vector_labels=False)
            raster_uri = crop_uri
            label_uri = label_crop_uri

        raster_source = RasterioSourceConfig(
            uris=[raster_uri], channel_order=channel_order)
        vector_source = GeoJSONVectorSourceConfig(
            uri=label_uri, default_class_id=4, ignore_crs_field=True)
        label_source = SemanticSegmentationLabelSourceConfig(
            raster_source=RasterizedSourceConfig(
                vector_source=vector_source,
                rasterizer_config=RasterizerConfig(background_class_id=3)))
        # Using with_rgb_class_map because label TIFFs have classes encoded as
        # RGB colors.
        # URI will be injected by scene config.
        # Using rgb=True innebär att prediktioner kommer ut i TIFF/RGB-format
        # Vektor output behöver läggas till nedan om ni tillför klasser, enligt class_id för nya klassen.
        # Background_class_id=3 och default_class_id=4 så nya klasser kan börja på 5.
        label_store = SemanticSegmentationLabelStoreConfig(
	rgb=True, 
	vector_output=[
		PolygonVectorOutputConfig(class_id=0), 
		PolygonVectorOutputConfig(class_id=1), 
		PolygonVectorOutputConfig(class_id=2),
        PolygonVectorOutputConfig(class_id=3),
        PolygonVectorOutputConfig(class_id=4)])

        scene = SceneConfig(
            id=id,
            raster_source=raster_source,
            label_source=label_source,
            label_store=label_store)

        return scene

    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[make_scene(id) for id in train_ids],
        validation_scenes=[make_scene(id) for id in val_ids])

    chip_sz = 300
    img_sz = chip_sz

    chip_options = SemanticSegmentationChipOptions(
        window_method=SemanticSegmentationWindowMethod.sliding, stride=chip_sz)

    if nochip:
        window_opts = {}
        # set window configs for training scenes
        for s in scene_dataset.train_scenes:
            window_opts[s.id] = GeoDataWindowConfig(
                method=GeoDataWindowMethod.sliding,
                size=chip_sz,
                stride=chip_options.stride)

        # set window configs for validation scenes
        for s in scene_dataset.validation_scenes:
            window_opts[s.id] = GeoDataWindowConfig(
                method=GeoDataWindowMethod.sliding,
                size=chip_sz,
                stride=chip_options.stride)

        data = SemanticSegmentationGeoDataConfig(
            scene_dataset=scene_dataset,
            window_opts=window_opts,
            img_sz=img_sz,
            img_channels=len(channel_order),
            num_workers=4,
            channel_display_groups=channel_display_groups,
            base_transform=base_transform,
            aug_transform=aug_transform,
            plot_options=PlotOptions(transform=plot_transform))
    else:
        data = SemanticSegmentationImageDataConfig(
            img_sz=img_sz,
            num_workers=4,
            channel_display_groups=channel_display_groups,
            base_transform=base_transform,
            aug_transform=aug_transform,
            plot_options=PlotOptions(transform=plot_transform))

    if external_model:
        model = SemanticSegmentationModelConfig(
            external_def=ExternalModuleConfig(
                github_repo='AdeelH/pytorch-fpn:0.2',
                name='fpn',
                entrypoint='make_fpn_resnet',
                entrypoint_kwargs={
                    'name': 'resnet50',
                    'fpn_type': 'panoptic',
                    'num_classes': len(class_config.names),
                    'fpn_channels': 256,
                    'in_channels': len(channel_order),
                    'out_size': (img_sz, img_sz)
                }))
    else:
        model = SemanticSegmentationModelConfig(backbone=Backbone.resnet50)

    backend = PyTorchSemanticSegmentationConfig(
        data=data,
        model=model,
        solver=SolverConfig(
            lr=1e-4,
            num_epochs=10,
            test_num_epochs=2,
            batch_sz=8,
            test_batch_sz=2,
            one_cycle=True),
        log_tensorboard=True,
        run_tensorboard=False,
        test_mode=test)

    pipeline = SemanticSegmentationConfig(
        root_uri=root_uri,
        dataset=scene_dataset,
        backend=backend,
        channel_display_groups=channel_display_groups,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz,
        chip_options=chip_options)

    return pipeline
